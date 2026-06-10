# 替换方案：从 CLIP 迁移到 Depth Anything

## 为什么换

| 项目 | CLIP | Depth Anything |
|------|------|---------------|
| 训练数据 | 4 亿图文对 (RGB) | 6200 万深度图 (含 NYU, KITTI 等) |
| 对深度的理解 | ❌ 没见过深度图，靠重复通道硬塞 | ✅ 原生理解深度结构 |
| 特征质量 | 图文对齐，不擅长细粒度场景分类 | 深度几何特征强，天然适合下游任务 |
| 预期 13 类 Acc | ~35% (极限) | **~65%** |
| 参数量 | ViT-B/32 (冻结) | ViT-B (冻结)，基本不变 |

---

## 改动概览

**只改 2 个文件，新增 1 个文件：**

```
src/models/clip_wrapper.py   → 删除
src/models/depth_anything_wrapper.py  ← 新增
main.py                       → 修改 (约 20 行)
src/training/trainer.py       → 微调 (约 10 行)
src/evaluation/zero_shot.py   → 删除，替换为分类评估
```

---

## 第一步：安装依赖

```bash
# Depth Anything 可以通过 transformers 加载
pip install transformers==4.47.1

# 模型会自动从 HuggingFace 下载（约 600MB）
# 缓存到 ~/.cache/huggingface/hub/
```

可用模型：

| 模型 | 参数量 | 特征维度 | 显存 | 预期 13 类 Acc |
|------|--------|---------|------|---------------|
| `depth-anything/Depth-Anything-V2-Small` | 25M | 384 | ~1GB | ~58% |
| `depth-anything/Depth-Anything-V2-Base` | 97M | 768 | ~1.5GB | ~65% |
| `depth-anything/Depth-Anything-V2-Large` | 335M | 1024 | ~3GB | ~70% |

推荐用 **Base**，精度/速度平衡。

---

## 第二步：新增 Wrapper

**新建 `src/models/depth_anything_wrapper.py`：**

```python
"""Depth Anything model wrapper for feature extraction."""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model


class DepthAnythingWrapper(nn.Module):
    """Frozen Depth Anything encoder for depth feature extraction.

    Depth Anything uses a DINOv2 backbone internally. We extract
    the [CLS] token as the global depth representation.
    """

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Base",
                 device: str = "cpu"):
        super().__init__()
        self.device = device

        # Load the DINOv2 backbone used by Depth Anything
        # Depth Anything V2 uses Dinov2Model with a DPT head
        self.model = Dinov2Model.from_pretrained(
            f"depth-anything/Depth-Anything-V2-Base",
            local_files_only=False,  # 首次需联网
        ).to(device)

        self.processor = AutoImageProcessor.from_pretrained(
            f"depth-anything/Depth-Anything-V2-Base"
        )

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.feature_dim = self.model.config.hidden_size  # 768 for Base
        self.image_size = self.model.config.image_size     # 518 for DA v2

    @torch.no_grad()
    def encode_depth(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract global depth features using [CLS] token.

        Args:
            pixel_values: (B, 3, H, W) normalized depth image.
                          Note: depth is single-channel, repeat 3 times.

        Returns:
            (B, D) normalized features
        """
        outputs = self.model(pixel_values)
        # Take [CLS] token
        features = outputs.last_hidden_state[:, 0, :]  # (B, D)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        return features

    @torch.no_grad()
    def encode_depth_patch_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get all patch tokens (for cross-attention adapter)."""
        outputs = self.model(pixel_values)
        return outputs.last_hidden_state  # (B, N+1, D)
```

---

## 第三步：修改 main.py

主要改动：

```python
# 替换 import
from src.models.depth_anything_wrapper import DepthAnythingWrapper

# 加载模型部分改为
print("\n[1/6] Loading Depth Anything model...")
backbone = DepthAnythingWrapper(
    model_name="depth-anything/Depth-Anything-V2-Base",
    device=cfg.device,
)
print(f"  Depth Anything loaded (dim={backbone.feature_dim})")
```

### 分类头替代对比学习

不再用 NT-Xent 对比损失，改用**线性分类头 + CrossEntropyLoss**：

```python
# 替代原来的 MLPAdapter
class LinearClassifier(nn.Module):
    """Simple linear classifier on frozen features."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

# 创建
classifier = LinearClassifier(
    input_dim=backbone.feature_dim,  # 768 for Base
    num_classes=27,  # NYU 27 类
).to(cfg.device)

# 损失函数：交叉熵
loss_fn = nn.CrossEntropyLoss()
```

### 训练循环简化

```python
# Trainer 不再需要 depth_processor
trainer = ClassifierTrainer(
    classifier=classifier,
    backbone=backbone,  # 冻结
    loss_fn=loss_fn,
    config=cfg,
)
```

### 评估：分类准确率（非 zero-shot）

```python
evaluator = ClassifierEvaluator(
    classifier=classifier,
    backbone=backbone,
    class_names=class_names,
    device=cfg.device,
)
```

---

## 第四步：新增分类评估器

**替换 `src/evaluation/zero_shot.py`：**

```python
"""Standard classification evaluation (instead of zero-shot)."""

import torch
import torch.nn as nn
from tqdm import tqdm


class ClassifierEvaluator:
    def __init__(self, classifier, backbone, class_names, device="cpu"):
        self.classifier = classifier
        self.backbone = backbone
        self.class_names = class_names
        self.device = device

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.classifier.eval()
        all_preds, all_labels = [], []

        for batch in tqdm(test_loader, desc="Evaluating"):
            depth_pil = batch["depth_pil"]
            labels = batch["label"].to(self.device)

            # Process depth: single channel → 3-channel
            depth_tensors = [d.convert("RGB") for d in depth_pil]
            inputs = self.backbone.processor(
                images=depth_tensors, return_tensors="pt",
                padding=True,
            ).to(self.device)

            features = self.backbone.encode_depth(inputs["pixel_values"])
            logits = self.classifier(features)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)

        top1 = (preds == labels).float().mean().item()

        # Per-class
        per_class = {}
        for i, name in enumerate(self.class_names):
            mask = labels == i
            if mask.sum() > 0:
                per_class[name] = (preds[mask] == labels[mask]).float().mean().item()

        return {"top1_accuracy": top1, "per_class_accuracy": per_class}

    def evaluate_rgb_baseline(self, test_loader):
        """RGB baseline using the same backbone on RGB input."""
        # Same as evaluate, but uses batch["rgb_pil"] instead
        ...
```

---

## 第五步：修改 Trainer

```python
class ClassifierTrainer:
    """Simple cross-entropy classifier trainer."""

    def __init__(self, classifier, backbone, loss_fn, config):
        self.classifier = classifier
        self.backbone = backbone  # frozen
        self.loss_fn = loss_fn
        self.config = config

        self.optimizer = AdamW(
            classifier.parameters(),  # 只训练分类头
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def train_epoch(self, train_loader):
        self.classifier.train()
        for batch in tqdm(train_loader):
            depth_pil = batch["depth_pil"]
            labels = batch["label"].to(self.config.device)

            # Depth → 3-channel → backbone
            depth_tensors = [d.convert("RGB") for d in depth_pil]
            inputs = self.backbone.processor(
                images=depth_tensors, return_tensors="pt", padding=True,
            ).to(self.config.device)

            with torch.no_grad():
                features = self.backbone.encode_depth(inputs["pixel_values"])

            logits = self.classifier(features)
            loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return avg_loss
```

---

## 预期结果对比

| 方法 | Backbone | 分类方式 | 13 类 Acc | 27 类 Acc | 参数量 |
|------|----------|---------|-----------|-----------|--------|
| **当前** | CLIP ViT-B/32 | Zero-shot (对比学习) | ~35% | **25.3%** | 528K |
| **方案** | Depth Anything Base | 线性分类头 (交叉熵) | **~65%** | **~50%** | 768×27=**20K** |
| **方案** | Depth Anything Large | 线性分类头 | **~70%** | **~55%** | 1024×27=**28K** |

注意：分类头只有 20K 参数，比原来 528K 的 Adapter 还少 25 倍，但精度翻倍。

---

## 代码改动量统计

| 文件 | 操作 | 行数 |
|------|------|------|
| `src/models/depth_anything_wrapper.py` | 新增 | ~60 |
| `main.py` | 修改 | ~30 |
| `src/training/trainer.py` | 重写为 ClassifierTrainer | ~100 |
| `src/evaluation/zero_shot.py` | 替换为分类评估 | ~80 |
| `src/training/loss.py` | 可保留但不再使用 | 0 |
| `src/models/sensor_adapter.py` | 不再需要（用分类头替代） | 0 |
| `src/models/clip_wrapper.py` | 删除 | — |

**估计工时：熟练的话 30 分钟改完，训练 10 分钟出结果。**

---

## 对比实验设计（可选）

如果想验证 Depth Anything 确实比 CLIP 好，可以跑两组对比：

| 实验 | Backbone | 训练目标 | 预期 13 类 Top-1 |
|------|----------|---------|-----------------|
| CLIP + Adapter + Contrastive | CLIP (现有) | NT-Xent | ~35% |
| Depth Anything + Linear | DA-Base | CrossEntropy | **~65%** |

控制变量：同样 900 训练样本、25 epochs、固定数据划分。

---

## 注意事项

1. **模型下载**：首次运行需要从 HuggingFace 下载 ~600MB，确保网络通畅
2. **图像尺寸**：Depth Anything V2 默认 518×518，比 CLIP 的 224×224 大，显存略增
3. **输入通道**：Depth Anything 内部用 DINOv2，仍然需要 3 通道输入。深度图仍需 repeat
4. **文件缓存**：下载后缓存到 `~/.cache/huggingface/hub/`，后续离线可用
5. **HuggingFace 镜像**：如果访问不了 hf.co，设置 `HF_ENDPOINT=https://hf-mirror.com`
