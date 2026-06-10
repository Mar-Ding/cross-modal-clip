"""Best-precision training script for Depth Anything + NYU13 scene classification.

Whats different from main.py:
- Maps 27 raw scene types → 13 semantic NYU classes (more data per class)
- Data augmentation: RandomHorizontalFlip, RandomAffine, ColorJitter
- Label smoothing loss
- 100 epochs with cosine annealing
- Best LR search via simple grid
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from pathlib import Path
import sys
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
from PIL import Image
from torchvision import transforms as T

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(__file__))

from src.models.depth_anything_wrapper import DepthAnythingWrapper
from src.data.nyu_mat_dataset import NYU13_CLASSES, NYU13_NAMES, NYUDepthMatDataset
from src.data.nyu_mat_dataset import create_dataloaders_from_mat
from src.evaluation.zero_shot import ClassifierEvaluator
from src.visualization.visualize import (
    plot_training_curves, plot_accuracy_comparison,
    plot_per_class_accuracy, save_results_json,
)

# ── Label smoothing loss ──────────────────────────────────────────────
class LabelSmoothCE(nn.Module):
    def __init__(self, epsilon=0.1, reduction="mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        smoothed = torch.full_like(logits, self.epsilon / (n_classes - 1))
        smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smoothed * log_probs).sum(dim=1)
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# ── Build 27→13 label mapping ────────────────────────────────────────
def build_nyu13_mapping(dataset):
    """Build mapping from 27 raw scene type IDs → 13 NYU class IDs."""
    mapping = {}
    for raw_id, aliased_name in dataset.type_names.items():
        if aliased_name in NYU13_CLASSES:
            mapping[raw_id] = NYU13_CLASSES[aliased_name]
        else:
            # Fallback: assign to first class
            print(f"  WARNING: '{aliased_name}' (raw_id={raw_id}) not in NYU13, mapping to 0")
            mapping[raw_id] = 0
    return mapping


def apply_label_map(batch, label_map):
    """Apply label mapping to a batch."""
    batch["label"] = torch.tensor(
        [label_map[int(l)] for l in batch["label"]],
        dtype=torch.long,
    )
    return batch


# ── Data loader with augmentation ────────────────────────────────────
def create_dataloaders_augmented(
    mat_path="nyu_depth_v2_labeled.mat",
    num_train=900, num_val=200, num_test=249,
    batch_size=16, image_size=518, num_workers=0, seed=42,
    augment=True,
):
    """Create dataloaders with augmentation for training set."""

    # Base transforms (resize + to tensor)
    base_transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=Image.BILINEAR),
        T.ToTensor(),
    ])

    # Augmentation for training
    if augment:
        train_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.92, 1.08),
                           fill=0),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
        ])
    else:
        train_transform = base_transform

    # Load full dataset
    full = NYUDepthMatDataset(
        mat_path=mat_path, split="all",
        image_size=image_size, seed=seed,
    )

    # Shuffle indices
    indices = list(range(len(full)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    total_needed = num_train + num_val + num_test
    if total_needed > len(indices):
        ratio = len(indices) / total_needed
        num_train = int(num_train * ratio)
        num_val = int(num_val * ratio)
        num_test = len(indices) - num_train - num_val

    train_idx = indices[:num_train]
    val_idx = indices[num_train:num_train + num_val]
    test_idx = indices[num_train + num_val:num_train + num_val + num_test]

    # Build label map from full dataset
    label_map = build_nyu13_mapping(full)
    num_classes = 13
    class_names = [NYU13_NAMES[i] for i in range(num_classes)]

    # Dataset wrapper with transform
    class AugmentedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform, label_map, is_train):
            self.subset = subset
            self.transform = transform
            self.label_map = label_map
            self.is_train = is_train

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            item = self.subset[idx]
            # Transform depth image
            depth_pil = item["depth_pil"].convert("RGB")
            if self.is_train:
                depth_tensor = self.transform(depth_pil)
            else:
                depth_tensor = self.transform(depth_pil)
            # Transform RGB for baseline
            rgb_pil = item["rgb_pil"].convert("RGB")
            if self.is_train:
                rgb_tensor = base_transform(rgb_pil)
            else:
                rgb_tensor = base_transform(rgb_pil)

            label = self.label_map[int(item["label"])]
            return {
                "depth_tensor": depth_tensor,
                "rgb_tensor": rgb_tensor,
                "label": label,
                "scene": item["scene"],
            }

    def collate_fn(batch):
        return {
            "depth_tensor": torch.stack([b["depth_tensor"] for b in batch]),
            "rgb_tensor": torch.stack([b["rgb_tensor"] for b in batch]),
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "scene": [b["scene"] for b in batch],
        }

    from torch.utils.data import Subset

    train_set = AugmentedSubset(Subset(full, train_idx), train_transform, label_map, is_train=True)
    val_set = AugmentedSubset(Subset(full, val_idx), base_transform, label_map, is_train=False)
    test_set = AugmentedSubset(Subset(full, test_idx), base_transform, label_map, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"  Classes: {num_classes} ({', '.join(class_names)})")
    print(f"  Label mapping: {len(label_map)} raw → {num_classes} NYU13")

    return train_loader, val_loader, test_loader, class_names


# ── Trainer ───────────────────────────────────────────────────────────
class BestTrainer:
    def __init__(self, classifier, backbone, loss_fn, config):
        self.classifier = classifier
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.config = config
        self.device = config["device"]
        self.use_rgb = config.get("use_rgb", False)

        self.optimizer = AdamW(
            classifier.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"]
        )

        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _extract(self, batch):
        if self.use_rgb:
            pixel_values = batch["rgb_tensor"]
        else:
            pixel_values = batch["depth_tensor"]
        # pixel_values is already (B, 3, H, W) tensor
        # We need to normalize via the backbone processor
        # But the processor expects PIL images... let's pass directly
        with torch.no_grad():
            # Direct encode since pixel_values are already resized
            features = self.backbone.encode_depth(pixel_values.to(self.device))
        return features

    def train_epoch(self, loader):
        self.classifier.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="Train", leave=False):
            labels = batch["label"].to(self.device)
            features = self._extract(batch)
            logits = self.classifier(features)
            loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(loader, desc="Val", leave=False):
            labels = batch["label"].to(self.device)
            features = self._extract(batch)
            logits = self.classifier(features)
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(loader), correct / total

    def train(self, train_loader, val_loader):
        best_acc = 0.0
        for epoch in range(self.config["epochs"]):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.scheduler.step()

            print(f"Ep {epoch+1:3d}/{self.config['epochs']} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.2%}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.classifier.state_dict(),
                           self.output_dir / "best_classifier.pt")

        torch.save(self.classifier.state_dict(), self.output_dir / "final_classifier.pt")
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Best Val Acc: {best_acc:.2%}")
        return self.history


def main():
    parser = argparse.ArgumentParser(
        description="Best-precision depth scene classification (NYU13)")
    parser.add_argument("--backbone", type=str, default="facebook/dinov2-base")
    parser.add_argument("--load-depth-weights", type=str, default="",
                        help="Path to Depth Anything .pth")
    parser.add_argument("--use-rgb", action="store_true")
    parser.add_argument("--mat-path", type=str, default="nyu_depth_v2_labeled.mat")
    parser.add_argument("--num-train", type=int, default=900)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=249)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--output-dir", type=str, default="./output_best_depth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {device}, Depth: {'RGB' if args.use_rgb else 'depth'}, "
          f"Augment: {not args.no_augment}")

    # ── 1. Backbone ──
    print(f"\n[1/4] Loading backbone: {args.backbone}")
    backbone = DepthAnythingWrapper(
        model_name=args.backbone,
        device=device,
        depth_weights_path=args.load_depth_weights or None,
    )

    # ── 2. Data ──
    print(f"\n[2/4] Loading NYU data (13-class mode)...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders_augmented(
        mat_path=args.mat_path,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        batch_size=args.batch_size,
        image_size=518,
        num_workers=0,
        seed=args.seed,
        augment=not args.no_augment,
    )
    num_classes = len(class_names)

    # ── 3. Classifier ──
    print(f"\n[3/4] Creating classifier ({num_classes} classes, "
          f"dim={backbone.feature_dim})...")
    classifier = nn.Linear(backbone.feature_dim, num_classes).to(device)
    params = sum(p.numel() for p in classifier.parameters())
    print(f"  Params: {params:,}")

    # ── 4. Train ──
    config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "device": device,
        "output_dir": args.output_dir,
        "use_rgb": args.use_rgb,
    }
    loss_fn = LabelSmoothCE(epsilon=args.label_smooth)
    trainer = BestTrainer(classifier, backbone, loss_fn, config)

    print(f"\n[4/4] Training {args.epochs} epochs...")
    history = trainer.train(train_loader, val_loader)
    plot_training_curves(history)

    # ── 5. Evaluate ──
    print(f"\nEvaluating on test set...")
    best_path = Path(args.output_dir) / "best_classifier.pt"
    if best_path.exists():
        classifier.load_state_dict(torch.load(best_path, map_location=device))

    # Test evaluation using our own loop
    classifier.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(test_loader, desc="Test"):
        labels = batch["label"].to(device)
        if args.use_rgb:
            px = batch["rgb_tensor"].to(device)
        else:
            px = batch["depth_tensor"].to(device)
        with torch.no_grad():
            features = backbone.encode_depth(px)
            logits = classifier(features)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    top1 = (preds == labels).float().mean().item()

    per_class = {}
    for i, name in enumerate(class_names):
        mask = labels == i
        if mask.sum() > 0:
            per_class[name] = (preds[mask] == labels[mask]).float().mean().item()
        else:
            per_class[name] = float("nan")

    print(f"\n  Test Top-1: {top1:.2%}")
    print(f"  Per-class:")
    for name, acc in sorted(per_class.items()):
        if not np.isnan(acc):
            print(f"    {name}: {acc:.2%}")

    # RGB baseline
    if not args.use_rgb:
        print(f"\n  RGB baseline...")
        classifier.eval()
        all_preds_rgb, all_labels_rgb = [], []
        for batch in tqdm(test_loader, desc="RGB"):
            labels = batch["label"].to(device)
            px = batch["rgb_tensor"].to(device)
            with torch.no_grad():
                features = backbone.encode_depth(px)
                logits = classifier(features)
            preds = logits.argmax(dim=1)
            all_preds_rgb.append(preds.cpu())
            all_labels_rgb.append(labels.cpu())
        preds_rgb = torch.cat(all_preds_rgb)
        labels_rgb = torch.cat(all_labels_rgb)
        rgb_top1 = (preds_rgb == labels_rgb).float().mean().item()
        print(f"  RGB Top-1: {rgb_top1:.2%}")

        # Save comparison
        results = {"top1_accuracy": top1, "per_class_accuracy": per_class}
        rgb_results = {"top1_accuracy": rgb_top1}
        plot_accuracy_comparison(results, rgb_results)
        save_results_json(results)
    else:
        results = {"top1_accuracy": top1, "per_class_accuracy": per_class}
        save_results_json(results)

    plot_per_class_accuracy(per_class)
    print(f"\nDone! Results in {args.output_dir}/ and output/")


if __name__ == "__main__":
    main()
