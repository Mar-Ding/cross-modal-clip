"""Main entry point for depth scene classification with DINOv2 backbone.

Usage:
    python main.py                                    # Default: NYU depth
    python main.py --use-rgb                          # RGB baseline
    python main.py --backbone dinov2-base             # Specify backbone
    python main.py --num-train 600 --epochs 25        # Tune training
    python main.py --load-depth-weights /path/to.pth  # Depth Anything weights
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import os
from pathlib import Path
import sys

# Set HF mirror for environments with restricted network
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.models.depth_anything_wrapper import DepthAnythingWrapper
from src.training.trainer import ClassifierTrainer
from src.evaluation.zero_shot import ClassifierEvaluator
from src.visualization.visualize import (
    plot_training_curves,
    plot_accuracy_comparison,
    plot_per_class_accuracy,
    save_results_json,
)
from src.data.nyu_mat_dataset import create_dataloaders_from_mat


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Depth Scene Classification with DINOv2 Backbone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["train", "evaluate", "visualize", "all"],
                        help="Pipeline mode")
    parser.add_argument("--backbone", type=str, default="facebook/dinov2-base",
                        choices=["facebook/dinov2-small", "facebook/dinov2-base",
                                 "facebook/dinov2-large"],
                        help="Backbone model")
    parser.add_argument("--load-depth-weights", type=str, default="",
                        help="Path to Depth Anything .pth file")
    parser.add_argument("--use-rgb", action="store_true",
                        help="Use RGB images instead of depth")
    parser.add_argument("--mat-path", type=str,
                        default="nyu_depth_v2_labeled.mat",
                        help="Path to NYU .mat file")
    parser.add_argument("--num-train", type=int, default=900,
                        help="Number of training samples")
    parser.add_argument("--num-val", type=int, default=200,
                        help="Number of validation samples")
    parser.add_argument("--num-test", type=int, default=249,
                        help="Number of test samples (max 249)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Check for .mat file
    mat_path = Path(args.mat_path)
    if not mat_path.exists():
        print(f"ERROR: .mat file not found at {mat_path}")
        print(f"  Download from: https://horatio.cs.nyu.edu/mit/silberman/")
        print(f"  nyu_depth_v2/nyu_depth_v2_labeled.mat")
        sys.exit(1)

    set_seed(args.seed)

    # Config
    cfg = Config(
        backbone_model_name=args.backbone,
        depth_weights_path=args.load_depth_weights,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_samples=args.num_train,
        num_val_samples=args.num_val,
        num_test_samples=args.num_test,
        output_dir=args.output_dir,
        seed=args.seed,
        use_rgb_input=args.use_rgb,
    )

    if args.use_rgb:
        print("Mode: RGB baseline (classifying RGB images)")
    else:
        print("Mode: Depth classification (classifying depth maps)")

    data_src = "NYU Depth V2 (.mat)"
    print(f"Config: device={cfg.device}, backbone={cfg.backbone_model_name}")
    print(f"  Data: {data_src}, train={cfg.num_train_samples}, "
          f"val={cfg.num_val_samples}, test={cfg.num_test_samples}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load backbone
    print(f"\n[1/5] Loading backbone model...")
    depth_weights = cfg.depth_weights_path if cfg.depth_weights_path else None
    backbone = DepthAnythingWrapper(
        model_name=cfg.backbone_model_name,
        device=cfg.device,
        depth_weights_path=depth_weights,
    )
    print(f"  Backbone loaded: {cfg.backbone_model_name} (dim={backbone.feature_dim})")

    # Determine number of classes from actual dataset
    print(f"\n[2/5] Loading dataset to determine class count...")
    full_dataset_for_info = None
    try:
        from src.data.nyu_mat_dataset import NYUDepthMatDataset
        temp = NYUDepthMatDataset(
            mat_path=str(mat_path), split="all",
            image_size=cfg.image_size, seed=cfg.seed,
        )
        num_classes = len(set(temp.scene_types.tolist()))
        class_names = [v for k, v in sorted(temp.type_names.items(), key=lambda x: x[0])]
        full_class_names = temp.full_class_names
        print(f"  Found {num_classes} scene types")
        for t in sorted(temp.type_names.keys()):
            cnt = int((temp.scene_types == t).sum())
            print(f"    [{t}] {temp.type_names[t]}: {cnt} samples")
        full_dataset_for_info = temp
    except Exception as e:
        print(f"  Could not inspect dataset: {e}, using default 13 classes")
        num_classes = 13
        class_names = cfg.nyu_classes
        full_class_names = class_names

    # Create linear classifier
    print(f"\n[3/5] Creating linear classifier ({num_classes} classes)...")
    classifier = nn.Linear(backbone.feature_dim, num_classes).to(cfg.device)
    print(f"  Classifier params: {sum(p.numel() for p in classifier.parameters()):,}")

    # Loss
    loss_fn = nn.CrossEntropyLoss()

    # Data loaders
    print(f"\n[4/5] Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders_from_mat(
        mat_path=str(mat_path),
        num_train=cfg.num_train_samples,
        num_val=cfg.num_val_samples,
        num_test=cfg.num_test_samples,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    if args.mode in ("train", "all"):
        print(f"  Train: {len(train_loader.dataset)} | "
              f"Val: {len(val_loader.dataset)} | "
              f"Test: {len(test_loader.dataset)}")

        print(f"\nTraining linear classifier...")
        trainer = ClassifierTrainer(classifier, backbone, loss_fn, cfg)
        history = trainer.train(train_loader, val_loader)
        plot_training_curves(history)

    if args.mode in ("evaluate", "all"):
        print(f"\n[5/5] Evaluating on test set...")

        best_path = output_dir / "best_classifier.pt"
        if best_path.exists():
            classifier.load_state_dict(torch.load(best_path, map_location=cfg.device))
            print(f"  Loaded best classifier from {best_path}")
        else:
            print(f"  WARNING: No saved classifier found at {best_path}")

        evaluator = ClassifierEvaluator(
            classifier, backbone, class_names,
            device=cfg.device, use_rgb=args.use_rgb,
        )
        results = evaluator.evaluate(test_loader)

        print(f"\n  Depth Results:")
        print(f"  Top-1 Accuracy: {results['top1_accuracy']:.2%}")
        print(f"  Per-class:")
        for name, acc in results["per_class_accuracy"].items():
            print(f"    {name}: {acc:.2%}")

        if not args.use_rgb:
            print(f"\n  Computing RGB baseline (DINOv2 upper bound)...")
            rgb_results = evaluator.evaluate_rgb_baseline(test_loader)
            print(f"  RGB Top-1: {rgb_results['top1_accuracy']:.2%}")
            plot_accuracy_comparison(results, rgb_results)
        else:
            rgb_results = None

        plot_per_class_accuracy(results["per_class_accuracy"])

        # Save results
        save_results_json(results)

    if args.mode in ("visualize",):
        print(f"\nGenerating visualizations from saved data...")
        if (output_dir / "results.json").exists():
            print("  Results already saved, see output/ directory.")
        else:
            print("  No saved results found. Run 'evaluate' mode first.")

    print("\nDone! Check output/ directory for results and visualizations.")


if __name__ == "__main__":
    main()
