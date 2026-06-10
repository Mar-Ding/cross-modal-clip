# CLIP 跨模态传感器适配 — Ubuntu 部署文档

## 1. 拉取代码

```bash
git clone https://github.com/Mar-Ding/cross-modal-clip.git
cd cross-modal-clip
```

## 2. 安装依赖

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets pillow matplotlib tqdm scikit-learn scipy
```

## 3. 下载数据集

NYU Depth V2 官方 .mat 文件（2.8GB）：

```bash
wget -O nyu_depth_v2_labeled.mat \
  "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
```

或使用 HuggingFace datasets（需要网络直连 HF）：

```python
python -c "from datasets import load_dataset; ds = load_dataset('nyu_depth_v2', split='train'); print(len(ds))"
```

## 4. 训练

**小规模（本地 CPU/GPU 快速验证）：**
```bash
python main.py --data synthetic --num-train 600 --num-val 100 --num-test 200 --epochs 30
```

**NYU 真实数据（推荐 600+ 样本）：**
```bash
python main.py --data mat --mat-path nyu_depth_v2_labeled.mat \
  --num-train 600 --num-val 100 --num-test 200 \
  --epochs 50 --batch-size 32 --lr 1e-3
```

**更多样本（全量 1449 张）：**
```bash
python main.py --data mat \
  --num-train 900 --num-val 200 --num-test 349 \
  --epochs 80 --batch-size 32
```

## 5. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | synthetic | synthetic / mat |
| `--mat-path` | nyu_depth_v2_labeled.mat | .mat 文件路径 |
| `--num-train` | 600 | 训练样本数 |
| `--num-val` | 100 | 验证样本数 |
| `--num-test` | 200 | 测试样本数 |
| `--batch-size` | 16 | 批次大小 |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 1e-3 | 学习率 |
| `--adapter` | mlp | mlp / cross_attn |
| `--output-dir` | ./output | 输出目录 |

## 6. 输出

训练完成后 `output/` 目录包含：
- `training_curves.png` — 损失+精度曲线
- `accuracy_comparison.png` — Depth vs RGB vs Random 精度对比
- `per_class_accuracy.png` — 13 类场景的每类精度
- `best_adapter.pt` — 最优模型权重
- `results.json` — 完整评估数据

## 7. 后续

结果出来后叫我做 PPT 展示。
GitHub 代码已最新：`github.com/Mar-Ding/cross-modal-clip`
