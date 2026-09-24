import os
import time

from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
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

DATA_DIR = "./data"        # 数据集存放目录，程序会下载MNIST到这里
BATCH_SIZE = 64            # 每次从数据加载器输入的样本数
EPOCHS = 20                # 训练的轮数
LR = 0.001                 # 学习率
HIDDEN1 = 32               # 第一个卷积层输出通道数
HIDDEN2 = 64               # 第二个卷积层输出通道数
HIDDEN3 = 128              # 第三个全连接层神经元个数
HIDDEN4 = 64               # 第四个全连接层神经元个数
torch.manual_seed(0)       # 设置随机种子，这样结果可以复现
NUM_WORKERS = 0 if WINDOWS else min(8, (os.cpu_count() or 2))   # 数据加载进程数: Windows 下保持单进程(和原来一致);
                                                                # Linux/WSL 下 fork 很快, 按核数开到 8
DRAW_DIR = "./draw"        # 这是存放损失曲线图像的文件地址


#准备数据集
def create_dataset():

    train_dataset = FashionMNIST(root=DATA_DIR, train=True, transform=ToTensor(), download=True)
    test_dataset = FashionMNIST(root=DATA_DIR, train=False, transform=ToTensor(), download=True)

    return train_dataset, test_dataset

#定义模型
class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        # 卷积块 1
        self.conv1 = nn.Conv2d(1, HIDDEN1, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积块 2
        self.conv2 = nn.Conv2d(HIDDEN1, HIDDEN2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(HIDDEN2 * 7 * 7, HIDDEN3)
        self.linear2 = nn.Linear(HIDDEN3, HIDDEN4)
        self.output3 = nn.Linear(HIDDEN4, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.output3(x)
        return x

#开始训练
def train(train_dataset):
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True)
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_history = []
    acc_history = []

    for epoch in range(EPOCHS):
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

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
        print(f'epoch: {epoch + 1}, loss: {total_loss / total_samples:.5f}, acc:{total_correct / total_samples:.4f}, time:{time.time() - start:.3f}s')
    torch.save(model.state_dict(), './model/level2_CNN_2.pth')   #保存数据

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
    plt.xticks(list(epochs_range))
    save_path = os.path.join(DRAW_DIR, 'CNN_DRAW_2.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"损失曲线已保存为 {save_path}")

#测试模型
def evaluate(test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = MLP().to(device)

    model.load_state_dict(torch.load('./model/level2_CNN_2.pth',map_location=device))
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