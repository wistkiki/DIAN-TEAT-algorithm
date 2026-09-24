import argparse
import json
import math
import os
import random
import shutil
import time

import numpy as np
import matplotlib

# ---------------- 平台开关 ----------------
# 关掉(WINDOWS = False) -> Windows 本机; 打开(WINDOWS = True) -> Linux / WSL
# 也可以用环境变量临时切换, 不用改代码:  set PLATFORM=windows  /  export PLATFORM=windows
PLATFORM = os.environ.get("PLATFORM", "windows" if os.name == "nt" else "linux").lower()
WINDOWS = PLATFORM.startswith("win")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
if WINDOWS:
    # Windows 自带 SimHei / Microsoft YaHei, 直接用这两个画中文
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
else:
    # Linux / WSL 没有上面两个字体, 需要先执行: sudo apt install -y fonts-noto-cjk
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False      # 不然负号会显示成方块
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

Image.MAX_IMAGE_PIXELS = None                   # 数据集里有大图, 关掉解压炸弹警告


# ========================= 超参数 =========================
DATA_ROOT = "./deli"                    # 原始数据: <日期>/dataset/{input,output}
CACHE_DIR = "./data/note_erasure"       # 缓存目录
DRAW_DIR = "./draw"
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "LV4_UNet_Erase.pth")

CACHE_SIZE = 768        # letterbox 后的缓存分辨率 (正方形)
CROP = 384              # 训练时随机裁剪的边长
BATCH = 6               # 每个 step 的样本数
MAX_STEPS = 3000        # 总共训练多少个 step
EVAL_EVERY = 100        # 每多少个 step 在全图上评估一次
LR = 3e-4               # 学习率 (图像回归任务常用 1e-4~3e-4, 之前用 1e-3 偏大)
WEIGHT_DECAY = 1e-5
WARMUP = 100            # 学习率预热步数
GRAD_CLIP = 1.0         # 梯度裁剪阈值
L1_W, SSIM_W = 1.0, 0.3 # 损失 = L1_W*L1 + SSIM_W*(1-SSIM)
BASE = 32               # U-Net 第一层通道数
AMP = True              # 混合精度
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_GT_INK = 0.0        # >0 时丢弃"GT 近似空白"的脏样本(设 0.02 可开启清洗)
SEED = 42
NUM_WORKERS = 8 if WINDOWS else min(8, (os.cpu_count() or 4))
BUDGET_YUAN = 30.0      # 服务器预算
CLOUD_PRICE = 1.5       # 云 GPU 租用价, 元/小时

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False   # 开着的话每个新输入尺寸都要重跑算法基准测试,
                                         # 逐张评估时会慢几百倍, 这里全程关掉
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve(p):
    """相对路径按本文件所在目录解析, 这样从任意目录运行都能找到数据"""
    return p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.abspath(__file__)), p)


def yuan_cost(seconds):
    """秒 -> 元 (云 GPU 租用口径)"""
    return seconds / 3600.0 * CLOUD_PRICE


# ========================= ① 数据准备与缓存 =========================
def to_gray(img):
    """统一转灰度。RGBA 直接 convert('L') 会把透明区域当成黑色,
    所以先按白底合成再转 —— 数据里有 24.6% 是 RGBA/P 模式。"""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("L")


def letterbox(img, size):
    """等比缩放到长边 = size, 再用白边填充成 size*size。

    为什么不直接 resize 成正方形: 把 1143x2000 拉成 768x768 会把文字纵向压扁 42%,
    笔画宽度分布被破坏, 网络学到的"笔迹先验"就失真了。"""
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def gt_ink(gpath):
    """GT 的墨迹密度 = 1 - 平均灰度, 用来识别"GT 近似空白"的非文档样本"""
    with Image.open(gpath) as im:
        im.draft("L", (256, 256))
        arr = np.asarray(im.convert("L").resize((256, 256), Image.BILINEAR), dtype=np.float32)
    return 1.0 - arr.mean() / 255.0


def scan_pairs(raw_root):
    """扫描 input/output 配对, 返回 [(group, input_path, output_path), ...]"""
    pairs = []
    for group in sorted(os.listdir(raw_root)):
        gdir = os.path.join(raw_root, group, "dataset")
        idir, odir = os.path.join(gdir, "input"), os.path.join(gdir, "output")
        if not (os.path.isdir(idir) and os.path.isdir(odir)):
            continue
        outs = {os.path.splitext(f)[0]: f for f in os.listdir(odir)}
        n0 = len(pairs)
        for f in sorted(os.listdir(idir)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            k = os.path.splitext(f)[0]
            if k not in outs:
                continue
            # 数据清洗: MIN_GT_INK 默认 0 = 全保留(和 dian 口径一致);
            # 设成 0.02 就会丢掉"GT 近似空白"的人物照片/手绘图这类脏样本
            if MIN_GT_INK > 0 and gt_ink(os.path.join(odir, outs[k])) < MIN_GT_INK:
                continue
            pairs.append((group, os.path.join(idir, f), os.path.join(odir, outs[k])))
        print(f"    {group}: {len(pairs) - n0} 对")
    return pairs


def prepare_cache(limit=0):
    """把所有图 letterbox + 灰度化后缓存成 .npy, 并做 80/10/10 划分。

    为什么缓存: 训练时零解码开销; npy 用 mmap 读, 1.4GB 数据不进内存,
    多个 DataLoader worker 共享同一份。"""
    raw_root = resolve(DATA_ROOT)
    out_dir = resolve(CACHE_DIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/4] 扫描配对: {raw_root}")
    pairs = scan_pairs(raw_root)
    print(f"      共 {len(pairs)} 对")
    if limit > 0:
        pairs = pairs[:limit]
        print(f"      --limit 生效, 只处理 {len(pairs)} 对")
    if not pairs:
        raise RuntimeError("没有找到任何配对样本")

    n = len(pairs)
    need = n * CACHE_SIZE * CACHE_SIZE * 2            # 两个 npy
    free = shutil.disk_usage(out_dir).free
    print(f"[2/4] 缓存需要约 {need / 1e9:.2f} GB, 磁盘可用 {free / 1e9:.2f} GB")
    if free < need * 1.1:
        raise RuntimeError(f"磁盘空间不足: 需要 {need / 1e9:.2f} GB, 只剩 {free / 1e9:.2f} GB")

    # 划分: 固定 seed 打乱后 80/10/10 (和 dian 口径一致, 便于横向对比)
    rng = random.Random(SEED)
    order = list(range(n))
    rng.shuffle(order)
    n_test, n_val = int(n * TEST_RATIO), int(n * VAL_RATIO)
    split_idx = {"test": order[:n_test],
                 "val": order[n_test:n_test + n_val],
                 "train": order[n_test + n_val:]}
    print(f"[3/4] 划分: train={len(split_idx['train'])} "
          f"val={len(split_idx['val'])} test={len(split_idx['test'])}")

    print(f"[4/4] letterbox -> {CACHE_SIZE}x{CACHE_SIZE} 并缓存为 .npy memmap")
    inp_path = os.path.join(out_dir, f"input_{CACHE_SIZE}.npy")
    tgt_path = os.path.join(out_dir, f"target_{CACHE_SIZE}.npy")
    inp = np.lib.format.open_memmap(inp_path, mode="w+", dtype=np.uint8,
                                    shape=(n, CACHE_SIZE, CACHE_SIZE))
    tgt = np.lib.format.open_memmap(tgt_path, mode="w+", dtype=np.uint8,
                                    shape=(n, CACHE_SIZE, CACHE_SIZE))

    # 按"划分内顺序"重排, 让同一个 split 的样本在 npy 里连续
    flat = split_idx["train"] + split_idx["val"] + split_idx["test"]
    offset = {orig: pos for pos, orig in enumerate(flat)}

    records = []
    t0 = time.time()
    for pos, orig in enumerate(flat):
        g, ip, op = pairs[orig]
        with Image.open(ip) as a:
            inp[pos] = letterbox(to_gray(a), CACHE_SIZE)
        with Image.open(op) as b:
            tgt[pos] = letterbox(to_gray(b), CACHE_SIZE)
        records.append({"key": os.path.splitext(os.path.basename(ip))[0], "group": g,
                        "input": ip.replace("\\", "/"), "target": op.replace("\\", "/")})
        if (pos + 1) % 200 == 0 or pos + 1 == n:
            el = time.time() - t0
            print(f"      {pos + 1}/{n}  ({el:.0f}s, {el / (pos + 1) * 1000:.0f} ms/对)", flush=True)
    inp.flush(); tgt.flush()
    del inp, tgt

    splits = {records[offset[o]]["key"]: name for name, lst in split_idx.items() for o in lst}
    meta = {"cache_size": CACHE_SIZE, "num_samples": n, "seed": SEED,
            "input_cache": os.path.basename(inp_path),
            "target_cache": os.path.basename(tgt_path),
            "split_sizes": {k: len(v) for k, v in split_idx.items()},
            "records": [dict(r, split=splits[r["key"]]) for r in records]}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"      完成, 耗时 {time.time() - t0:.0f}s -> {out_dir}")


# ========================= ② 数据集与增强 =========================
class _Gamma:
    """随机 gamma 校正: 模拟不同曝光下纸张亮度的非线性"""

    def __init__(self, lo=0.75, hi=1.35):
        self.lo, self.hi = lo, hi

    def __call__(self, x):
        return x.clamp(1e-6, 1.0).pow(float(torch.empty(1).uniform_(self.lo, self.hi)))


class _GaussianNoise:
    """加性高斯噪声: 模拟高 ISO 拍摄的噪点"""

    def __init__(self, sigma=0.03):
        self.sigma = sigma

    def __call__(self, x):
        return x + torch.randn_like(x) * self.sigma


class NoteErasureDataset(Dataset):
    """读取 .npy 缓存, 训练时做文档增强。

    **关键设计: 只对 input 做增强, target 永远保持干净。**
    在监督式图像修复里 target 是要逼近的真值; 对它做光度抖动等于让网络去学
    "干净图 + 随机噪声"这个自相矛盾的目标, 训练会发散或退化成均值图。
    """

    def __init__(self, root, split="train", crop_size=CROP, augment=False):
        self.root, self.crop_size, self.augment = root, crop_size, augment
        with open(os.path.join(root, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.size = self.meta["cache_size"]
        self.inputs = np.load(os.path.join(root, self.meta["input_cache"]), mmap_mode="r")
        self.targets = np.load(os.path.join(root, self.meta["target_cache"]), mmap_mode="r")
        recs = self.meta["records"]
        self.records = recs if split == "all" else [r for r in recs if r["split"] == split]
        if not self.records:
            raise ValueError(f"split={split!r} 没有样本")
        index = {r["key"]: i for i, r in enumerate(recs)}      # records 顺序 == npy 顺序
        self.indices = [index[r["key"]] for r in self.records]
        self._build_aug()

    def _build_aug(self):
        if not self.augment:
            self.geom = self.photo = None
            return
        self.geom = T.Compose([
            # 随机尺度 + 裁剪: 模拟拍摄距离差异(0.6x~1.4x), 让模型见过各种字号
            T.RandomResizedCrop(size=(self.crop_size, self.crop_size), scale=(0.36, 1.0),
                                ratio=(0.85, 1.18), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.RandomRotation(degrees=3.0)], p=0.5),                       # 纸张摆放角度
            T.RandomApply([T.RandomPerspective(distortion_scale=0.06, p=1.0)], p=0.3),   # 镜头畸变
        ])
        self.photo = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.28, contrast=0.28)], p=0.9),
            T.RandomApply([_Gamma()], p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.3),     # 对焦不准
            T.RandomApply([_GaussianNoise(0.03)], p=0.3),
            T.RandomErasing(p=0.15, scale=(0.005, 0.04), ratio=(0.5, 2.0), value=1.0),   # 反光/遮挡
        ])

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        """多进程 DataLoader 会把 dataset pickle 给 worker。

        numpy 的 memmap 默认是**按值** pickle 的 —— 实测整个 dataset 序列化出来 2.8 GB,
        塞进进程间管道会被截断, worker 直接起不来(pickle data was truncated)。
        这里改成只传路径, 到 worker 里再重新打开 mmap(仍然是共享映射, 不会复制数据)。
        注意: dian 项目是靠把 num_workers 默认设成 0 来绕开这个问题的, 一旦调大就会踩到。"""
        st = self.__dict__.copy()
        st["inputs"] = st["targets"] = None
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self.inputs = np.load(os.path.join(self.root, self.meta["input_cache"]), mmap_mode="r")
        self.targets = np.load(os.path.join(self.root, self.meta["target_cache"]), mmap_mode="r")

    def __getitem__(self, i):
        idx = self.indices[i]
        # 注意: inputs 是只读 memmap, astype 会复制一份可写数组, 不能省
        inp = torch.from_numpy(self.inputs[idx].astype(np.float32) / 255.0).unsqueeze(0)
        tgt = torch.from_numpy(self.targets[idx].astype(np.float32) / 255.0).unsqueeze(0)

        if self.augment:
            inp, tgt = self._paired_geom(inp, tgt)
            inp = self.photo(inp).clamp(0, 1)
        elif self.crop_size:
            inp, tgt = self._center_crop_pair(inp, tgt, self.crop_size)
        return inp, tgt

    def _paired_geom(self, inp, tgt):
        """几何变换必须对 input 和 target 用**同一组参数**, 否则两者空间上不再对应,
        等于给网络喂了错误标签。做法: 沿通道维拼起来过一次变换, 再拆开。"""
        both = self.geom(torch.cat([inp, tgt], dim=0))
        a, b = both[:inp.shape[0]], both[inp.shape[0]:]
        if a.shape != b.shape:                                  # 极端情况兜底
            b = F.interpolate(b.unsqueeze(0), size=a.shape[-2:], mode="bilinear",
                              align_corners=False).squeeze(0)
        return a, b

    @staticmethod
    def _center_crop_pair(inp, tgt, size):
        """验证/测试: 从中心裁剪(确定性, 保证指标可比)"""
        h, w = inp.shape[-2:]
        if h < size or w < size:
            inp = F.pad(inp, (0, max(0, size - w), 0, max(0, size - h)), value=1.0)
            tgt = F.pad(tgt, (0, max(0, size - w), 0, max(0, size - h)), value=1.0)
            h, w = inp.shape[-2:]
        top, left = (h - size) // 2, (w - size) // 2
        return (inp[:, top:top + size, left:left + size],
                tgt[:, top:top + size, left:left + size])


def build_loaders(cache_dir, crop=CROP, batch=BATCH, workers=NUM_WORKERS, limit=0):
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    tr = NoteErasureDataset(cache_dir, "train", crop, augment=True)
    if limit > 0:                                   # 小数据验证流程
        tr.records, tr.indices = tr.records[:limit], tr.indices[:limit]
    va = NoteErasureDataset(cache_dir, "val", crop_size=0, augment=False)    # 整图
    te = NoteErasureDataset(cache_dir, "test", crop_size=0, augment=False)
    return DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True, **kw), va, te


# ========================= ③ 模型 =========================
class DoubleConv(nn.Module):
    """U-Net 的基本单元: 两次 3*3 卷积 + BatchNorm + ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """下采样: MaxPool 后接 DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """上采样 + skip 拼接 + DoubleConv。

    用双线性插值上采样而不是转置卷积: 转置卷积容易产生棋盘格伪影,
    对"擦除笔迹"这种要求平滑输出的任务尤其致命, 而且插值上采样没有参数。"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:     # 处理奇数分辨率下采样后的 1 像素误差
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """灰度 1 通道进 / 1 通道出。

    手写笔迹既需要"看清局部笔画"(浅层高分辨率特征), 也需要"理解这是不是印刷体/表格线"
    (深层语义特征), 跳跃连接同时提供两者, 避免下采样时丢掉细笔画。"""

    def __init__(self, in_ch=1, out_ch=1, base=BASE, depth=4, out_act="sigmoid"):
        super().__init__()
        if depth < 2:
            raise ValueError("depth 至少为 2")
        self.depth, self.out_act = depth, out_act
        chans = [base * (2 ** i) for i in range(depth)]

        self.inc = DoubleConv(in_ch, base)
        self.downs = nn.ModuleList([Down(chans[i], chans[i + 1]) for i in range(depth - 1)])
        self.bottleneck = Down(chans[-1], chans[-1] * 2)

        ups = []
        for i in range(depth - 1, -1, -1):
            in_ch_up = chans[i] * 2 if i == depth - 1 else chans[i + 1]
            ups.append(Up(in_ch_up, chans[i], chans[i]))
        self.ups = nn.ModuleList(ups)
        self.outc = nn.Conv2d(base, out_ch, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = []
        x = self.inc(x); skips.append(x)
        for d in self.downs:
            x = d(x); skips.append(x)
        x = self.bottleneck(x)
        for i, up in enumerate(self.ups):
            x = up(x, skips[self.depth - 1 - i])
        x = self.outc(x)
        return torch.sigmoid(x) if self.out_act == "sigmoid" else x


# ========================= ④ 指标与损失 =========================
def psnr(pred, target, max_val=1.0):
    """逐图 PSNR, 返回 shape=(N,)"""
    mse = (pred - target).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(max_val ** 2 / mse)


def _gaussian_kernel(size=11, sigma=1.5, device=None):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    return (g.t() @ g).unsqueeze(0).unsqueeze(0)


def ssim(pred, target, max_val=1.0, kernel_size=11):
    """逐图 SSIM, 返回 shape=(N,)。口径和 skimage 的
    structural_similarity(gaussian_weights=True, sigma=1.5, use_sample_covariance=False) 一致。
    带梯度, 所以既能当指标也能当损失。"""
    c = pred.shape[1]
    k = _gaussian_kernel(kernel_size, 1.5, pred.device).repeat(c, 1, 1, 1)
    pad = kernel_size // 2
    mu1 = F.conv2d(pred, k, padding=pad, groups=c)
    mu2 = F.conv2d(target, k, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(pred * pred, k, padding=pad, groups=c) - mu1_sq
    s2 = F.conv2d(target * target, k, padding=pad, groups=c) - mu2_sq
    s12 = F.conv2d(pred * target, k, padding=pad, groups=c) - mu12
    C1, C2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    smap = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return smap.flatten(1).mean(dim=1)


class CombinedLoss(nn.Module):
    """L1 + SSIM 的加权和。

    单用 L1/L2 都不直接优化 PSNR/SSIM; SSIM 项管局部结构/对比度, 对"这条笔画该不该擦"
    这种结构判断很关键, 也能缓解模型输出"平摊的灰"。"""

    def __init__(self, l1_w=L1_W, ssim_w=SSIM_W):
        super().__init__()
        self.l1_w, self.ssim_w = l1_w, ssim_w

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        if self.ssim_w > 0:
            return self.l1_w * l1 + self.ssim_w * (1.0 - ssim(pred, target).mean())
        return self.l1_w * l1


def degenerate_ratio(pred, thr=0.995):
    """退化检测: 输出几乎全白或全黑的比例。
    只看 PSNR 会被"直接输出白纸"这种作弊解骗过去, 所以必须单独检查。"""
    white = (pred > 0.98).float().mean()
    black = (pred < 0.02).float().mean()
    return 1.0 if (white > thr or black > thr) else 0.0


@torch.no_grad()
def full_image_metrics(model, ds, tag="val", limit=0):
    """在**全图**(letterbox 后的完整缓存图)上算 PSNR/SSIM —— 对外汇报一律用这个口径。

    为什么不用训练时的 384 小块: 局部方差小会让 PSNR 系统性偏高。"""
    model.eval()
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    ps, ss, dg = [], [], []
    for i in range(n):
        inp, tgt = ds[i]
        inp = inp.unsqueeze(0).to(device)
        tgt = tgt.unsqueeze(0).to(device)
        pred = model(inp)
        h = min(pred.size(2), tgt.size(2)); w = min(pred.size(3), tgt.size(3))
        pred, tgt = pred[:, :, :h, :w], tgt[:, :, :h, :w]
        ps.append(psnr(pred, tgt).item())
        ss.append(ssim(pred, tgt).item())
        dg.append(degenerate_ratio(pred[0]))
    return {"tag": tag, "n": n, "psnr": float(np.mean(ps)),
            "ssim": float(np.mean(ss)), "degenerate": float(np.mean(dg))}


# ========================= ⑤ 训练 =========================
def lr_at(step, max_steps, base_lr, warmup):
    """warmup + 余弦衰减"""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, p)))


def train(args):
    cache_dir = resolve(CACHE_DIR)
    if not os.path.exists(os.path.join(cache_dir, "meta.json")):
        print("缓存不存在, 先构建...")
        prepare_cache(limit=args.limit)

    os.makedirs(resolve(DRAW_DIR), exist_ok=True)
    os.makedirs(resolve(MODEL_DIR), exist_ok=True)
    max_steps = args.max_steps
    use_amp = AMP and device.type == "cuda"

    train_loader, val_ds, test_ds = build_loaders(cache_dir, limit=args.limit)
    model = UNet().to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"设备: {device}")
    print(f"U-Net 参数量: {n_param:,} ({n_param / 1e6:.2f} M)   输入通道: 1 (灰度)")
    print(f"缓存 {train_loader.dataset.size}x{train_loader.dataset.size}   "
          f"train={len(train_loader.dataset)} val={len(val_ds)} test={len(test_ds)}")
    print(f"损失 {L1_W}*L1 + {SSIM_W}*(1-SSIM)   lr={LR} warmup={WARMUP} "
          f"grad_clip={GRAD_CLIP} AMP={use_amp}")

    criterion = CombinedLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step_loss, ev_step, ev_loss, ev_psnr, ev_ssim = [], [], [], [], []
    data_iter = iter(train_loader)
    start = time.time()
    best_ssim = -1.0
    print("-" * 96)

    for step in range(1, max_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        cur_lr = lr_at(step - 1, max_steps, LR, WARMUP)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        model.train()
        with torch.amp.autocast("cuda", enabled=use_amp):
            y_pred = model(x)
        # 损失必须在 autocast **外面**算(即强制 fp32):
        # SSIM 要计算方差 E[x^2]-E[x]^2, 也就是两个 ~0.9 的数相减得到 ~1e-3 —— fp16 只有
        # 约 3 位有效数字, 结果会变成 0 或负数, 再除以 C2=(0.03)^2 就出 NaN。
        # 实测: 在 autocast 里算 SSIM 损失时梯度全是 nan, GradScaler 每步把 scale 减半并
        # 跳过参数更新, 模型一步都没学到(loss 恒为 0.40, 验证 SSIM 反而从 0.71 掉到 0.51)。
        # 前向留在 autocast 里, 照样享受 AMP 的速度。
        loss = criterion(y_pred.float(), y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)   # 防梯度尖峰带偏训练
        scaler.step(optimizer)
        scaler.update()
        step_loss.append(loss.item())

        if step % EVAL_EVERY == 0 or step == 1:
            m = full_image_metrics(model, val_ds, "val", limit=args.eval_n)
            elapsed = time.time() - start
            ev_step.append(step)
            ev_loss.append(float(np.mean(step_loss[-EVAL_EVERY:])))
            ev_psnr.append(m["psnr"])
            ev_ssim.append(m["ssim"])
            print(f"step {step:5d}/{max_steps} | loss {ev_loss[-1]:.5f} | lr {cur_lr:.2e} | "
                  f"全图 PSNR {m['psnr']:6.2f} dB | SSIM {m['ssim']:.4f} | 退化 {m['degenerate']:.3f} | "
                  f"{elapsed:5.0f}s | 云租 {yuan_cost(elapsed):.4f}元")
            if m["ssim"] > best_ssim:
                best_ssim = m["ssim"]
                torch.save({"model": model.state_dict(), "step": step,
                            "full_psnr": m["psnr"], "full_ssim": m["ssim"]}, resolve(MODEL_PATH))

    train_time = time.time() - start
    print("-" * 96)
    print(f"训练结束: {train_time:.0f}s ({train_time / 60:.1f} 分钟), 云租 {yuan_cost(train_time):.4f} 元, "
          f"占预算 {yuan_cost(train_time) / BUDGET_YUAN * 100:.3f}% (预算 {BUDGET_YUAN:.0f} 元)")

    ck = torch.load(resolve(MODEL_PATH), map_location=device)
    model.load_state_dict(ck["model"])
    print(f"已加载验证集最优权重 (step {ck['step']}, val SSIM {ck['full_ssim']:.4f})")

    plot_curves(ev_step, ev_loss, step_loss, ev_psnr, ev_ssim)
    save_demos(model, test_ds, n=4)
    res = full_image_metrics(model, test_ds, "test", limit=args.eval_n)
    print(f"[测试集] {res['n']} 张 | PSNR {res['psnr']:.2f} dB | SSIM {res['ssim']:.4f} | "
          f"退化比例 {res['degenerate']:.4f}")
    measure_cost(model, train_time, res)
    return res


# ========================= ⑥ 出图 / 成本 / 推理 =========================
def plot_curves(ev_step, ev_loss, step_loss, ev_psnr, ev_ssim):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(range(1, len(step_loss) + 1), step_loss, color="tab:red", lw=0.8, alpha=0.45,
               label="每 step")
    ax[0].plot(ev_step, ev_loss, "o-", color="darkred", ms=3,
               label=f"每 {EVAL_EVERY} step 平均")
    ax[0].set_xlabel("Step"); ax[0].set_ylabel("Loss (L1 + %.1f*(1-SSIM))" % SSIM_W)
    ax[0].set_title("Training Loss"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[1].plot(ev_step, ev_loss, "o-", color="darkred", ms=4)
    ax[1].set_xlabel("Step"); ax[1].set_ylabel("Loss")
    ax[1].set_title(f"Training Loss (每 {EVAL_EVERY} step)"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(resolve(DRAW_DIR), "UNet_LOSS.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Loss 曲线已保存为 {p}")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    l1, = ax1.plot(ev_step, ev_psnr, "o-", color="tab:blue", ms=3, label="PSNR (左轴)")
    ax1.set_xlabel("Step"); ax1.set_ylabel("PSNR (dB)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    l2, = ax2.plot(ev_step, ev_ssim, "s-", color="tab:green", ms=3, label="SSIM (右轴)")
    ax2.set_ylabel("SSIM", color="tab:green"); ax2.tick_params(axis="y", labelcolor="tab:green")
    ax1.legend(handles=[l1, l2], loc="lower right")
    ax1.set_title(f"PSNR / SSIM vs GT (全图口径, 每 {EVAL_EVERY} step)")
    fig.tight_layout()
    p = os.path.join(resolve(DRAW_DIR), "UNet_PSNR_SSIM.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"PSNR/SSIM 曲线已保存为 {p}")


@torch.no_grad()
def save_demos(model, ds, n=4):
    model.eval()
    step = max(1, len(ds) // (n + 1))
    for k in range(min(n, len(ds))):
        inp, tgt = ds[k * step]
        pred = model(inp.unsqueeze(0).to(device))[0, 0].cpu().numpy()
        fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))
        ax[0].imshow(inp[0].numpy(), cmap="gray", vmin=0, vmax=1); ax[0].set_title("输入 (带手写)")
        ax[1].imshow(pred, cmap="gray", vmin=0, vmax=1); ax[1].set_title("预测 (擦除后)")
        ax[2].imshow(tgt[0].numpy(), cmap="gray", vmin=0, vmax=1); ax[2].set_title("GT (干净)")
        for z in ax:
            z.axis("off")
        fig.tight_layout()
        p = os.path.join(resolve(DRAW_DIR), f"UNet_demo_{k + 1}.png")
        fig.savefig(p, dpi=110); plt.close(fig)
        print(f"对比图已保存为 {p}")


@torch.no_grad()
def measure_cost(model, train_time, test_res):
    """预测成本: 参数量 / FLOPs / 推理耗时 / 峰值显存 / 人民币 / 预算占比"""
    model.eval()
    L = []
    n_param = sum(p.numel() for p in model.parameters())
    L.append(f"参数量            : {n_param:,} ({n_param / 1e6:.2f} M)")
    try:
        from torch.utils.flop_counter import FlopCounterMode
        fc = FlopCounterMode(display=False)
        with fc:
            model(torch.randn(1, 1, CROP, CROP, device=device))
        L.append(f"计算量            : {fc.get_total_flops() / 1e9:.2f} GFLOPs (单张 {CROP}x{CROP})")
    except Exception:
        pass

    ms_list = []
    for n in (CROP, CACHE_SIZE):
        d = torch.randn(1, 1, n, n, device=device)
        for _ in range(5):
            model(d)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            model(d)
        torch.cuda.synchronize()
        ms = (time.time() - t) / 20 * 1000
        ms_list.append(ms)
        L.append(f"单张推理 {n}x{n:<4d}    : {ms:7.2f} ms/张   ({1000 / ms:6.1f} 张/秒)")

    torch.cuda.reset_peak_memory_stats()
    model(torch.randn(1, 1, CACHE_SIZE, CACHE_SIZE, device=device))
    torch.cuda.synchronize()
    L.append(f"单张推理峰值显存  : {torch.cuda.max_memory_allocated() / 1024 ** 2:.1f} MB")
    L.append(f"训练总耗时        : {train_time:.0f} s ({MAX_STEPS} 个 step, batch={BATCH}, crop={CROP})")
    L.append("-" * 44)
    L.append(f"人民币花费估算（云 GPU {CLOUD_PRICE} 元/小时）")
    L.append(f"训练一次          : {yuan_cost(train_time):.4f} 元")
    L.append(f"推理单张 {CACHE_SIZE}      : {yuan_cost(ms_list[-1] / 1000):.7f} 元")
    L.append(f"推理 1 万张       : {yuan_cost(ms_list[-1] / 1000 * 10000):.4f} 元")
    L.append(f"成本控制 预算 {BUDGET_YUAN:.0f} 元: 本次占 {yuan_cost(train_time) / BUDGET_YUAN * 100:.3f}%, "
             f"可训练约 {BUDGET_YUAN / yuan_cost(train_time):.0f} 次")
    L.append("-" * 44)
    L.append(f"测试集结果({test_res['n']} 张, 全图 {CACHE_SIZE}x{CACHE_SIZE} letterbox 口径)")
    L.append(f"  PSNR {test_res['psnr']:.2f} dB | SSIM {test_res['ssim']:.4f} | "
             f"退化比例 {test_res['degenerate']:.4f}")

    txt = "\n".join(L)
    print("\n===== 预测成本 =====")
    print(txt)
    p = os.path.join(resolve(DRAW_DIR), "UNet_cost.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("U-Net 手写擦除 —— 预测成本报告\n" + "=" * 44 + "\n" + txt + "\n")
    print(f"成本报告已保存为 {p}")


def load_model():
    m = UNet().to(device)
    ck = torch.load(resolve(MODEL_PATH), map_location=device)
    m.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    m.eval()
    return m


@torch.no_grad()
def evaluate_only(args):
    _, val_ds, test_ds = build_loaders(resolve(CACHE_DIR), workers=0)
    model = load_model()
    for tag, ds in (("val", val_ds), ("test", test_ds)):
        r = full_image_metrics(model, ds, tag, limit=args.eval_n)
        print(f"[{tag}] {r['n']} 张 | PSNR {r['psnr']:.2f} dB | SSIM {r['ssim']:.4f} | "
              f"退化 {r['degenerate']:.4f}")
    save_demos(model, test_ds, n=4)


@torch.no_grad()
def predict(image_path, out_path=None):
    """对单张任意尺寸的原图做擦除(灰度 + letterbox 到缓存分辨率)"""
    model = load_model()
    with Image.open(image_path) as im:
        g = letterbox(to_gray(im), CACHE_SIZE)
    x = torch.from_numpy(g.astype(np.float32) / 255.0)[None, None].to(device)
    pred = model(x)[0, 0].cpu().numpy()
    if out_path:
        Image.fromarray((pred * 255).astype(np.uint8)).save(out_path)
        print(f"已保存 {out_path}")
    return pred


# ========================= 主程序 =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Level 4: 基于 U-Net 的手写笔记擦除")
    ap.add_argument("--prepare", action="store_true", help="只构建数据缓存, 不训练")
    ap.add_argument("--eval", action="store_true", help="只评估已有权重")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只用前 N 对(小数据验证流程)")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--eval-n", type=int, default=60, help="每次全图评估用多少张验证图")
    ap.add_argument("--predict", metavar="IMG", help="对单张图做擦除")
    args = ap.parse_args()

    if args.prepare:
        prepare_cache(limit=args.limit)
    elif args.predict:
        predict(args.predict, os.path.join(resolve(DRAW_DIR), "UNet_predict.png"))
    elif args.eval:
        evaluate_only(args)
    else:
        train(args)
