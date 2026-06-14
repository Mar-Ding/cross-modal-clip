# 深度图场景分类：深度预训练特征 vs 通用自监督特征

基于 DINOv2 的深度图场景分类（NYU Depth V2 数据集），
比较普通 DINOv2 自监督权重与 Depth Anything V2 深度预训练权重的迁移学习效果。

## 核心发现

| 实验 | 骨干网络初始化 | 数据增强 | 13类验证精度 |
|:----:|:-------------:|:--------:|:----------:|
| C | DINOv2（普通自监督） | ✓ | **55.5%** |
| A | Depth Anything V2（深度预训练） | ✓ | 50.5% |
| — | DINOv2（无增强，25轮） | ✗ | 35.0% |

**结论**：预训练目标的对齐比领域对齐更重要 — DINOv2（语义特征）55.5% > Depth Anything（几何特征）50.5%。

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- 单 GPU（推荐 8GB+ 显存）或 CPU

## 安装

```bash
pip install -r requirements.txt
```

首次运行会自动从 HuggingFace 下载 DINOv2 模型（约 600MB 缓存到 `~/.cache/huggingface/hub/`）。
国内用户可通过环境变量使用 HF 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 数据集

使用 **NYU Depth V2** 官方标注子集（1449 对 RGB-深度图）。
数据集自动通过 HuggingFace Datasets 加载（`nyu_depth_v2`），无需手动下载。

## 使用方式

```bash
# 默认运行：DINOv2-Base + 深度图 + 数据增强 + 100轮
python main.py

# RGB 基线对比
python main.py --use-rgb

# 加载 Depth Anything V2 权重
python main.py --load-depth-weights /path/to/depth_anything_v2_vitb.pth

# 关闭数据增强（复现 35.0% 基线）
python main.py --no-augment --epochs 25 --lr 1e-3

# 指定骨干网络
python main.py --backbone facebook/dinov2-large
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|:----|:-----:|:----|
| `--backbone` | `facebook/dinov2-base` | 骨干网络（small/base/large） |
| `--load-depth-weights` | `""` | Depth Anything .pth 权重路径 |
| `--use-rgb` | — | 使用 RGB 图像替代深度图 |
| `--num-train` | 900 | 训练样本数 |
| `--num-val` | 200 | 验证样本数 |
| `--num-test` | 249 | 测试样本数 |
| `--batch-size` | 16 | 批大小 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 2e-3 | 学习率 |
| `--label-smooth` | 0.1 | 标签平滑系数 |
| `--no-augment` | — | 关闭数据增强 |
| `--output-dir` | `./output` | 输出目录 |

## 项目结构

```
depth-scene-classification/
├── main.py                    # 主入口（训练+评估+可视化）
├── src/
│   ├── config.py              # 配置管理
│   ├── data/
│   │   └── nyu_mat_dataset.py # NYU Depth V2 数据集加载器
│   ├── models/
│   │   └── depth_anything_wrapper.py  # DINOv2 骨干网络封装
│   ├── evaluation/
│   │   └── zero_shot.py       # 分类评估器
│   └── visualization/
│       └── visualize.py       # 训练曲线与结果可视化
├── paper/
│   ├── main.tex / main.pdf    # 英文论文
│   ├── main_cn.tex / main_cn.pdf # 中文论文
│   └── figures/               # 论文图表
├── requirements.txt           # 依赖
└── README.md                  # 本文件
```

## 输出

运行后 `--output-dir` 目录（默认 `./output/`）生成：

- `best_classifier.pt` — 最佳验证精度的模型权重
- `final_classifier.pt` — 最终模型权重
- `history.json` — 训练历史（loss + 精度）
- `training_curves.png` — 训练曲线图
- `accuracy_comparison.png` — 精度对比柱状图
- `per_class_accuracy.png` — 各类别精度图
- `results.json` — 测试集评估结果

## 论文

- **英文**: `paper/main.pdf`（IEEE 双栏，5 页）
- **中文**: `paper/main_cn.pdf`（IEEE 双栏，4 页）

## 引用

```bibtex
@misc{ding2026depth,
  author = {丁驿},
  title = {Depth Map Scene Classification: Are Depth-Pretrained
           Features Better Than Generic Self-Supervised Features?},
  year = {2026},
  school = {武汉理工大学},
}
```
