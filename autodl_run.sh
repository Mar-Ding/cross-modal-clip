#!/bin/bash
# AutoDL 启动脚本 - Depth Anything NYU13 消融实验
# 上传整个 cross-modal-clip 目录后运行:
#   bash autodl_run.sh
set -e

echo "=========================================="
echo "  Depth Anything NYU13 Ablation Study"
echo "=========================================="
echo ""

# ── 0. 路径配置 ──
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"
echo "Working directory: $REPO_DIR"
OUTPUT_BASE="./ablation_outputs"
mkdir -p "$OUTPUT_BASE"

# ── 1. 安装依赖 ──
echo ""
echo "[1/6] Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.1 h5py pillow matplotlib tqdm numpy
pip install huggingface_hub

# ── 2. 检查 GPU ──
echo ""
echo "[2/6] Checking GPU..."
python -c "
import torch
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else '')
"

# ── 3. 下载 NYU Depth V2 数据 ──
echo ""
echo "[3/6] Checking NYU Depth V2 data..."
if [ ! -f "nyu_depth_v2_labeled.mat" ]; then
    echo "  Downloading NYU Depth V2 (2.8GB)..."
    pip install gdown 2>/dev/null
    # NYU Depth V2 labeled.mat from NYU official
    wget -q --show-progress "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat" \
        -O nyu_depth_v2_labeled.mat 2>&1 || {
        echo "  WARNING: Direct download failed, trying alternative..."
        # Alternative: Google Drive mirror (user needs to provide)
        echo "  Please upload nyu_depth_v2_labeled.mat manually"
        exit 1
    }
else
    echo "  Found existing nyu_depth_v2_labeled.mat"
fi

# ── 4. 缓存 Depth Anything V2 权重（通过镜像） ──
echo ""
echo "[4/6] Pre-caching Depth Anything V2 weights..."
python -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download, snapshot_download
# Download DINOv2 model
print('  Downloading facebook/dinov2-base...')
from transformers import Dinov2Model, AutoImageProcessor
model = Dinov2Model.from_pretrained('facebook/dinov2-base')
proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
print('  Downloading facebook/dinov2-small...')
model_s = Dinov2Model.from_pretrained('facebook/dinov2-small')
proc_s = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
# Download Depth Anything .pth
print('  Downloading Depth-Anything-V2-Base weights...')
pth = hf_hub_download('depth-anything/Depth-Anything-V2-Base', 'depth_anything_v2_vitb.pth')
print(f'  Cached at: {pth}')
" 2>&1
DA_WEIGHTS=$(python -c "
import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from huggingface_hub import hf_hub_download
p = hf_hub_download('depth-anything/Depth-Anything-V2-Base', 'depth_anything_v2_vitb.pth')
print(p)
")

echo "  Depth Anything weights: $DA_WEIGHTS"

# ── 5. 消融实验 ──
echo ""
echo "[5/6] Running ablation experiments..."
echo ""

run_exp() {
    local name="$1"
    shift
    echo "────────────────────────────────────────"
    echo "  Experiment: $name"
    echo "  Args: $@"
    echo "────────────────────────────────────────"
    python train_best.py \
        --output-dir "$OUTPUT_BASE/$name" \
        "$@"
    echo ""
}

# ---- 实验 A: 核心配置（Depth + DA权重 + 增强）----
# 这就是最优配置，跑满100 epochs
run_exp "A_depth_da_aug" \
    --load-depth-weights "$DA_WEIGHTS" \
    --epochs 100 --lr 2e-3 --batch-size 16

# ---- 实验 B: 无数据增强 ----
run_exp "B_depth_da_noaug" \
    --load-depth-weights "$DA_WEIGHTS" \
    --no-augment \
    --epochs 100 --lr 2e-3 --batch-size 16

# ---- 实验 C: 普通 DINOv2（无 Depth Anything 权重）+ 增强 ----
run_exp "C_depth_dinov2_aug" \
    --epochs 100 --lr 2e-3 --batch-size 16

# ---- 实验 D: 普通 DINOv2 + 无增强（基线）----
run_exp "D_depth_dinov2_noaug" \
    --no-augment \
    --epochs 100 --lr 2e-3 --batch-size 16

# ---- 实验 E: RGB 基线（Depth 权重 + 增强）----
run_exp "E_rgb_da_aug" \
    --load-depth-weights "$DA_WEIGHTS" \
    --use-rgb \
    --epochs 100 --lr 2e-3 --batch-size 16

# ---- 实验 F: 小 backbone (dinov2-small) ----
run_exp "F_depth_small_da_aug" \
    --backbone "facebook/dinov2-small" \
    --load-depth-weights "$DA_WEIGHTS" \
    --epochs 100 --lr 2e-3 --batch-size 32

# ---- 实验 G: 不同学习率 1e-3 ----
run_exp "G_depth_da_aug_lr1e3" \
    --load-depth-weights "$DA_WEIGHTS" \
    --lr 1e-3 \
    --epochs 100 --batch-size 16

# ── 6. 汇总结果 ──
echo ""
echo "[6/6] Results summary..."
echo ""

python -c "
import json, os, glob

results = {}
for exp_dir in sorted(os.listdir('$OUTPUT_BASE')):
    hist_path = os.path.join('$OUTPUT_BASE', exp_dir, 'history.json')
    result_path = os.path.join('$OUTPUT_BASE', exp_dir, '..', '..', 'output', 'results.json')
    alt_path = './output/results.json'
    
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            h = json.load(f)
        best_val = max(h['val_acc']) if h['val_acc'] else 0
        final_val = h['val_acc'][-1] if h['val_acc'] else 0
        
        # Try to get test acc from results.json
        test_acc = None
        for candidate in [result_path, alt_path]:
            if os.path.exists(candidate):
                try:
                    with open(candidate) as f:
                        r = json.load(f)
                    test_acc = r.get('top1_accuracy', None)
                except:
                    pass
        
        results[exp_dir] = {
            'best_val': best_val * 100,
            'final_val': final_val * 100,
            'test_acc': test_acc * 100 if test_acc else None,
        }

print()
print('=' * 65)
print(f'  {\"Experiment\":<30s} {\"Best Val\":>10s} {\"Final Val\":>10s} {\"Test\":>10s}')
print('=' * 65)
for name, r in sorted(results.items()):
    tv = f\"{r['test_acc']:.1f}%\" if r['test_acc'] is not None else 'N/A'
    print(f'  {name:<30s} {r[\"best_val\"]:>9.1f}% {r[\"final_val\"]:>9.1f}% {tv:>10s}')
print('=' * 65)
"

echo ""
echo "=========================================="
echo "  All experiments complete!"
echo "  Results in: $OUTPUT_BASE/"
echo "  Summary above."
echo "=========================================="
