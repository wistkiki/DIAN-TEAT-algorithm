"""
用 ResNet 识别 FashionMNIST。

以 LV3_AlexNet.py 为基础改写: 数据加载 / 训练 / 测试 / 数据增强的框架完全保留, 只把模型换成 ResNet。
原版 ResNet 的输入是 224*224 彩色图, 第一层用 7*7、stride=2 的卷积再加上 3*3 最大池化,
28*28 的小图经不起这样连着两次下采样, 所以按 CIFAR 版的常规做法:
换成 3*3、stride=1 的卷积, 并且去掉那个最大池化层。

通道数取的是原版 ResNet-18 的一半（32/64/128/256）:
原版 64/128/256/512 有 11.17M 参数, 对 6 万张的 FashionMNIST 明显过剩, 实测每轮要 23 秒;
减半后约 2.8M 参数, 每轮只要 6~7 秒, 90 轮从 35 分钟降到约 10 分钟。
想还原原版宽度, 把 HIDDEN1~HIDDEN4 改回 64/128/256/512 即可。
"""

import os
import time

from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Compose, RandomCrop, RandomHorizontalFlip
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------- 平台开关 ----------------
# 关掉(WINDOWS = False) -> Windows 本机; 打开(WINDOWS = True) -> Linux / WSL
# 也可以用环境变量临时切换, 不用改代码:  set PLATFORM=windows  /  export PLATFORM=windows
PLATFORM = os.environ.get("PLATFORM", "windows" if os.name == "nt" else "linux").lower()
WINDOWS = PLATFORM.startswith("win")

DATA_DIR = "./data"        # 数据集存放目录，程序会下载FashionMNIST到这里
BATCH_SIZE = 64            # 每次从数据加载器输入的样本数
EPOCHS = 90                # 训练的轮数
LR = 0.01                  # 学习率（SGD 要用比 Adam 更大的学习率）
MOMENTUM = 0.9             # SGD 动量
WEIGHT_DECAY = 5e-4        # 权重衰减
HIDDEN1 = 32               # 第 1 个 stage 的输出通道数（原版 ResNet-18 是 64）
HIDDEN2 = 64               # 第 2 个 stage 的输出通道数（原版是 128）
HIDDEN3 = 128              # 第 3 个 stage 的输出通道数（原版是 256）
HIDDEN4 = 256              # 第 4 个 stage 的输出通道数（原版是 512）
NUM_BLOCKS = 2             # 每个 stage 里残差块的个数（2 个即 ResNet-18）
torch.manual_seed(0)       # 设置随机种子，这样初始权重一致
torch.backends.cudnn.benchmark = True   # 输入尺寸永远是 28*28, 让 cuDNN 自动挑最快的卷积算法
NUM_WORKERS = 8 if WINDOWS else min(8, (os.cpu_count() or 4))   # 数据加载的并行进程数(本机 32 个逻辑核心, 取 8 足够;
                                                                # Linux/WSL 下按实际核数取, 但不超过 8 以省共享内存)
DRAW_DIR = "./draw"        # 这是存放损失曲线图像的文件地址


#准备数据集
def create_dataset():
    # 训练集做数据增强: 先 pad 到 36*36 再随机裁回 28*28(相当于随机平移), 以及随机水平翻转。
    # 增强必须写在 ToTensor() 前面, 因为这些操作作用在 PIL 图片上, 张量上做不了。
    train_transform = Compose([
        RandomCrop(28, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
    ])
    # 测试集不用增强, 只用 ToTensor(), 否则准确率会掉。
    test_transform = ToTensor()

    train_dataset = FashionMNIST(root=DATA_DIR, train=True, transform=train_transform, download=True)
    test_dataset = FashionMNIST(root=DATA_DIR, train=False, transform=test_transform, download=True)

    return train_dataset, test_dataset

#定义残差块
class BasicBlock(nn.Module):
    """ResNet 的基本残差块: 两个 3*3 卷积, 再和输入相加(捷径 shortcut)"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 卷积后面接了 BatchNorm, 所以卷积自己的 bias 可以省掉
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径: 主路如果改变了尺寸或通道数, 捷径也要用 1*1 卷积跟着变, 否则没法相加
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()      # 尺寸通道都一致, 直接恒等映射

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)             # 这一步就是"残差连接"
        out = torch.relu(out)
        return out

#定义模型
class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 主干: 1*28*28 --conv(3*3)--> HIDDEN1*28*28
        self.conv1 = nn.Conv2d(1, HIDDEN1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(HIDDEN1)
        # 四个 stage, 每个 stage 的第 1 个残差块用 stride=2 把特征图尺寸减半
        self.layer1 = self.make_layer(HIDDEN1, HIDDEN1, NUM_BLOCKS, stride=1)   # 28*28
        self.layer2 = self.make_layer(HIDDEN1, HIDDEN2, NUM_BLOCKS, stride=2)   # 14*14
        self.layer3 = self.make_layer(HIDDEN2, HIDDEN3, NUM_BLOCKS, stride=2)   # 7*7
        self.layer4 = self.make_layer(HIDDEN3, HIDDEN4, NUM_BLOCKS, stride=2)   # 4*4
        # 全局平均池化: 把 HIDDEN4*4*4 压成 HIDDEN4*1*1, 这样就不需要很大的全连接层了
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.output = nn.Linear(HIDDEN4, 10)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        """拼一个 stage: 第 1 个块负责改变尺寸/通道, 后面几个块尺寸不变"""
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.output(x)
        return x

#开始训练
def train(train_dataset):
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)
    model = ResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # 学习率衰减: 每 30 轮把学习率乘以 0.1, 即 90 轮里 0.01 -> 0.001 -> 0.0001
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_history = []
    acc_history = []

    for epoch in range(EPOCHS):
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            model.train()
            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()
            total_loss += loss.item() * len(y)
            total_samples += len(y)

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)
        #打印结果
        print(f'epoch: {epoch + 1}, loss: {total_loss / total_samples:.5f}, acc:{total_correct / total_samples:.4f}, lr:{optimizer.param_groups[0]["lr"]:.5f}, time:{time.time() - start:.3f}s')
        scheduler.step()        # 每个 epoch 走一次, 到第 30/60 轮结束时学习率降为 1/10
                                # 放在 print 后面, 打印出来的 lr 才是本轮真正用的那个
    torch.save(model.state_dict(), './model/LV3_ResNet.pth')   #保存数据

#绘制损失曲线图像
    plt.figure(figsize=(10, 5))
    epochs_range = range(1, len(loss_history) + 1)
    os.makedirs(DRAW_DIR, exist_ok=True)
    plt.plot(epochs_range, loss_history, label='Training Loss', color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()
    plt.xticks(range(1, EPOCHS + 1, 5))     # 90 轮, 每 5 轮标一个刻度, 全标会糊成一片
    save_path = os.path.join(DRAW_DIR, 'ResNet_DRAW.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"损失曲线已保存为 {save_path}")

#测试模型
def evaluate(test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = ResNet().to(device)

    model.load_state_dict(torch.load('./model/LV3_ResNet.pth', map_location=device))
    total_correct, total_samples = 0, 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        model.eval()

        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=-1)
        total_correct += (y_pred == y).sum()
        total_samples += len(y)

    print(f'Acc: {total_correct / total_samples:.5f}')



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    train_dataset, test_dataset = create_dataset()
    #train(train_dataset)
    evaluate(test_dataset)
    pass
