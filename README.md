# 从 MLP 到 U-Net：深度学习逐级实践

> 2026 团队秋招算法题的个人解答 —— 五个难度层级，从全连接网络一路做到 U-Net 手写笔记擦除。

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu128-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)

---

## 这是什么

按照题目给的五个层级逐级实现，**每一级都配一份 README 记录思路、超参数、逐轮训练日志**。

---

## 关于ai的使用

题目中的五个部分，前两个即level0和1主要为手搓，后续的level2，3均以前面的MLP模型为基准，在ai的辅助下进行了改动，LEVEL4最初在ai辅助下一点点写，但是效果很差，然后主要代码以及level4的readme都是ai负责完成，仅有一些较为粗浅的对ai代码的个人理解写在学习记录中。

另外，本项目最早在Windows上跑，最后在ai帮助下为所有文件加上了Linux调整器，主要就是对中文解释器与多通道设置进行调整，有关Linux的对比也在学习记录中。

本项目无论在wsl还是Windows上都是在另外的虚拟环境下跑，这样可以有效避免污染主系统的环境。

关于lv0-4的README以及学习记录，都是由ai将docx文件转成markdown文件，初稿保存的就是原docx文件。


---

## 结果一览

| 层级 | 内容 | 数据集 | 主要代码 | 参数量 | 结果 |
| :--- | :--- | :--- | :--- | ---: | :--- |
| LEVEL0 | 环境自检 | —— | `LEVEL0.py` | —— | GPU 矩阵乘法比 CPU 快两个数量级 |
| LEVEL1 | MLP | MNIST | `LV1_MLP.py` | 235,146 | Acc **97.75%** |
| LEVEL1 | MLP | FashionMNIST | `LV1_MLP_2.py` | 235,146 | Acc **89.15%** |
| LEVEL2 | CNN | MNIST | `LV2_CNN.py` | 429,258 | Acc **98.86%** |
| LEVEL2 | CNN | FashionMNIST | `LV2_CNN_2.py` | 429,258 | Acc **91.63%** |
| LEVEL3 | AlexNet | FashionMNIST | `LV3_AlexNet.py` | 4,027,594 | Acc **93.47%** |
| LEVEL3 | ResNet | FashionMNIST | `LV3_ResNet.py` | 2,797,034 | Acc **95.12%** |
| LEVEL4 | **U-Net 手写擦除** | 自建试卷数据集 | `LV4_UNet_Erase.py` | 7,849,025 | PSNR **25.86 dB** / SSIM **0.9640** |

> LEVEL4 的详细分析、基线对照和成本报告见 [`README_LV4UNet.md`](README_LV4UNet.md)。

### LEVEL4 的关键数字

| 项目 | 数值 |
| :--- | :--- |
| 数据 | 2412 对（input 带手写 / output 干净 GT），80/10/10 随机划分 = 1930 / 241 / 241 |
| 测试集 | PSNR **25.86 dB**，SSIM **0.9640**，退化样本比例 **0.0000** |
| 基线对照 | 原样输出 11.12 dB / 全白输出 15.98 dB —— **模型比最强作弊解高 9.88 dB** |
| 训练成本 | 3000 step，**445 秒**，按云 GPU 1.5 元/小时折算 **0.1853 元**（预算 30 元，占 0.618%） |
| 推理成本 | 768×768 单张 **38.61 ms**（25.9 张/秒），处理 1 万张约 0.16 元 |
| 峰值显存 | 1067.9 MB |

**「退化比例 = 0.0000」是硬指标**：题目要求"不出现全白、全黑"，我没有靠肉眼看几张图，而是把这条验收标准做成了可自动监控的量（单图超过 98% 像素 > 0.98 判为全白），测试集 241 张全程为 0。

---

## 文件结构

```
dian2/
├── LEVEL0.py                环境检查：CPU / GPU 算力对比（含热身与同步）
├── LV1_MLP.py               MLP  → MNIST
├── LV1_MLP_2.py             MLP  → FashionMNIST
├── LV2_CNN.py               CNN  → MNIST
├── LV2_CNN_2.py             CNN  → FashionMNIST
├── LV3_AlexNet.py           AlexNet → FashionMNIST
├── LV3_ResNet.py            ResNet  → FashionMNIST
├── LV4_UNet_Erase.py        U-Net 手写笔记擦除 ★
│
├── README_LV0.md            ┐
├── README_LV1.md            │
├── README_LV2.md            │ 每一级的完整报告：
├── README_LV3AlexNet.md     │ 网络结构 / 超参数 / 逐轮日志 / 踩坑
├── README_LV3ResNet.md      │
├── README_LV4UNet.md        ┘
├── 学习记录.md               逐级学习记录与问题反思
├── requirement.txt          环境配置记录（Windows + WSL 双平台）
│
├── draw/                    训练产物：损失曲线、PSNR/SSIM 曲线、擦除效果对比图
├── model/                   训练好的权重（.pth）
├── 初稿/                     题目原始草稿（含题目 PDF）
├── data/                    数据集与缓存 —— 不入库
└── deli/                    原始试卷图片 —— 不入库
```

---

## 快速开始

```bash
# 1) 环境
conda create -n pytorch python=3.10 -y && conda activate pytorch
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy matplotlib pillow

# 2) 跑任意一级（MNIST / FashionMNIST 会自动下载）
python LV1_MLP.py          # 或 LV2_CNN_2.py、LV3_ResNet.py ...

# 3) LEVEL4 分三步
python LV4_UNet_Erase.py --prepare                  # 建数据缓存（约 8-10 分钟）
python LV4_UNet_Erase.py --limit 64 --max-steps 100 # 小数据先验证流程（几十秒）
python LV4_UNet_Erase.py                            # 正式训练（约 7.4 分钟）
```

**详细环境说明（含 WSL 双平台配置、`sm_120` 架构注意事项）见 [`requirement.txt`](requirement.txt)。**

---

## 两个设计约定

**1. 所有脚本都带平台开关。** Windows 和 Linux/WSL 双平台可跑，默认自动判断，也可以用环境变量强制指定：

```python
PLATFORM = os.environ.get("PLATFORM", "windows" if os.name == "nt" else "linux").lower()
WINDOWS = PLATFORM.startswith("win")
```

它控制**中文字体**（Linux 用 Noto CJK，否则图里中文变方框）和 **DataLoader 进程数**。实测两种设置下 6 个脚本各跑 1 个 epoch 全部通过。

**2. LV4 的指标口径是"全图 letterbox"，并且明确标注水分。** PSNR/SSIM 是全参考指标，必须用整张图算才有意义（384×384 小块算出来的 PSNR 会系统性偏高）。同时我在报告里如实扣除：768 画布中 47.8% 是恒定白边，白拿约 **+2.82 dB**。**只报绝对 PSNR 而不报基线，在这个任务上等于自欺欺人。**

---


## 环境

| | Windows 本机 | WSL (Ubuntu 24.04) |
| :--- | :--- | :--- |
| Python | 3.10.21 | 3.12.3 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 | 同左 |
| CUDA Runtime / cuDNN | 12.8 / 9.19.0 | 同左 |
| GPU | RTX 5060 Laptop 8 GB（sm_120） | 同左 |

> RTX 5060 是 Blackwell 架构（**sm_120**），必须 PyTorch ≥ 2.7 的官方 wheel。装老了会报 `no kernel image is available`。验证：`any('120' in a for a in torch.cuda.get_arch_list())` 必须为 `True`。

---

## 说明

- 数据集与训练缓存不包含在本仓库中（`data/` 约 2.86 GB、`deli/` 约 1.8 GB、权重约 61 MB）。MNIST / FashionMNIST 由脚本自动下载；试卷数据集需自行获取。
- LEVEL4 的模型权重如需复现结果，可重新训练：3000 step 约 7.4 分钟、成本约 0.19 元。
- 固定 `SEED = 42` 保证**数据划分一致**，但由于 cuDNN 非位级确定 + 多进程增强随机流不同步，重新训练会有小幅浮动，demo 图不会逐像素一致。
