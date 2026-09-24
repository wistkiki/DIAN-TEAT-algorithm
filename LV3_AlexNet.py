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
EPOCHS = 90                # 训练的轮数（原版 AlexNet 是 90 轮）
LR = 0.01                  # 学习率（SGD 要用比 Adam 更大的学习率）
MOMENTUM = 0.9             # SGD 动量（原版 AlexNet 的值）
WEIGHT_DECAY = 5e-4        # 权重衰减（原版 AlexNet 的值）
HIDDEN1 = 64               # 第一个卷积层输出通道数
HIDDEN2 = 192              # 第二个卷积层输出通道数
HIDDEN3 = 384              # 第三个卷积层输出通道数
HIDDEN4 = 256              # 第四个卷积层输出通道数
HIDDEN5 = 256              # 第五个卷积层输出通道数
HIDDEN6 = 1024             # 第一个全连接层神经元个数
HIDDEN7 = 512              # 第二个全连接层神经元个数
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

#定义模型
class AlexNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 卷积块 1: 1*28*28 --conv(5*5)--> HIDDEN1*28*28 --pool(3*3,s2)--> HIDDEN1*13*13
        self.conv1 = nn.Conv2d(1, HIDDEN1, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积块 2: HIDDEN1*13*13 --conv(5*5)--> HIDDEN2*13*13 --pool(3*3,s2)--> HIDDEN2*6*6
        self.conv2 = nn.Conv2d(HIDDEN1, HIDDEN2, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积块 3/4/5: 三个 3*3 卷积连着堆, 尺寸不变, 只换通道
        self.conv3 = nn.Conv2d(HIDDEN2, HIDDEN3, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(HIDDEN3, HIDDEN4, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(HIDDEN4, HIDDEN5, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)      # HIDDEN5*6*6 -> HIDDEN5*2*2
        # 全连接头
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)                          # AlexNet 的标志之一: 全连接层前 dropout
        self.linear1 = nn.Linear(HIDDEN5 * 2 * 2, HIDDEN6)
        self.linear2 = nn.Linear(HIDDEN6, HIDDEN7)
        self.output3 = nn.Linear(HIDDEN7, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool3(torch.relu(self.conv5(x)))
        x = self.flatten(x)
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.dropout(torch.relu(self.linear2(x)))
        x = self.output3(x)
        return x

#开始训练
def train(train_dataset):
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)
    model = AlexNet().to(device)
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
    torch.save(model.state_dict(), './model/LV3_AlexNet.pth')   #保存数据

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
    save_path = os.path.join(DRAW_DIR, 'AlexNet_DRAW.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"损失曲线已保存为 {save_path}")

#测试模型
def evaluate(test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = AlexNet().to(device)

    model.load_state_dict(torch.load('./model/LV3_AlexNet.pth', map_location=device))
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
