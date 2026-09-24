"""
本文件用于测试本机环境
"""

import os
import time

import torch


# ---------------- 平台开关 ----------------
# 关掉(WINDOWS = False) -> Windows 本机; 打开(WINDOWS = True) -> Linux / WSL
# 也可以用环境变量临时切换, 不用改代码:  set PLATFORM=windows  /  export PLATFORM=windows
PLATFORM = os.environ.get("PLATFORM", "windows" if os.name == "nt" else "linux").lower()
WINDOWS = PLATFORM.startswith("win")

print(f"当前平台: {PLATFORM} ({'Windows 本机' if WINDOWS else 'Linux / WSL'})")
print(f"torch 版本: {torch.__version__}")
print(f"torch 编译时带的 cuda 版本: {torch.version.cuda}")
print(f"cuda 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"显卡名称: {torch.cuda.get_device_name(0)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"本次使用的设备: {device}")


def matmul_time(dev, n=2000, repeat=5):
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)

    for _ in range(3):              # 热身, 显卡第一次计算要先初始化, 不计入耗时
        _ = a @ b
    if dev.type == "cuda":
        torch.cuda.synchronize()    # GPU 是异步执行的, 必须同步一下才能拿到真实耗时

    start = time.time()
    for i in range(repeat):
         a @ b
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - start) / repeat


N = 4096
cpu_time = matmul_time(torch.device("cpu"), N)
print(f"cpu平均耗时: {cpu_time * 1000:.1f} ms")

if torch.cuda.is_available():
    gpu_time = matmul_time(torch.device("cuda"), N)
    print(f"gpu平均耗时: {gpu_time * 1000:.1f} ms")