# Flower Photos — YOLO11 花卉分类

基于 YOLO11 的五类花卉图像分类项目。

## 数据集

5 类花卉：雏菊（daisy）、蒲公英（dandelion）、玫瑰（roses）、向日葵（sunflowers）、郁金香（tulips）。

```
dataset/
├── train/          # 训练集（80%）
└── val/            # 验证集（20%）
```

## 环境

- Python 3.8+
- PyTorch + CUDA（推荐 GPU 训练）
- [Ultralytics](https://github.com/ultralytics/ultralytics) `pip install ultralytics`
- scikit-learn、Pillow

## 快速开始

### 1. 准备数据

```bash
python prepare_data.py
```

将原始分类文件夹（daisy/、dandelion/、roses/、sunflowers/、tulips/）按 8:2 划分复制到 `dataset/train/` 和 `dataset/val/`。

### 2. 训练模型

```bash
python train.py
```

使用 `yolo11n-cls.pt` 预训练权重，训练参数：

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 224×224 |
| Epochs | 50 |
| Batch Size | 16 |
| 学习率 | 0.001 |
| 早停 | patience=10 |
| 设备 | CUDA |

### 3. 运行 Demo

```bash
python demo/demo.py
```

基于 Tkinter 的 GUI 花卉分类器，选择图片后自动预测并显示中文结果。

## 项目结构

```
flower_photos/
├── dataset/              # 数据集（train/val 切分）
├── demo/demo.py          # Tkinter GUI 分类器
├── prepare_data.py       # 数据预处理脚本
├── train.py              # YOLO11 训练脚本
├── yolo11n-cls.pt        # 预训练权重
└── runs/                 # 训练输出（gitignore）
```

## 模型

使用 YOLO11n-cls 分类模型，基于 ImageNet 预训练权重微调。
