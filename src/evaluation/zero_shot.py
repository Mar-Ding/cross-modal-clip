"""Standard classification evaluation using linear head on frozen features."""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Optional
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class ClassifierEvaluator:
    """Evaluate linear classifier accuracy on test set.

    Pipeline:
    1. Depth/RGB image → backbone → frozen features
    2. Features → Linear classifier → logits
    3. argmax → Top-1 accuracy
    """

    def __init__(
        self,
        classifier: nn.Module,
        backbone: nn.Module,
        class_names: List[str],
        device: str = "cpu",
        use_rgb: bool = False,
    ):
        self.classifier = classifier
        self.backbone = backbone
        self.class_names = class_names
        self.device = device
        self.use_rgb = use_rgb

    def _extract_features(self, batch):
        """Extract features from depth (or RGB) images using frozen backbone."""
        if self.use_rgb:
            pil_images = batch["rgb_pil"]
        else:
            pil_images = batch["depth_pil"]

        rgb_tensors = [d.convert("RGB") for d in pil_images]
        inputs = self.backbone.processor(
            images=rgb_tensors,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            features = self.backbone.encode_depth(inputs["pixel_values"])
        return features

    @torch.no_grad()
    def evaluate(self, test_loader) -> dict:
        """Run evaluation on test set.

        Returns:
            dict with accuracy metrics and per-class breakdown
        """
        self.classifier.eval()

        all_preds = []
        all_labels = []

        for batch in tqdm(test_loader, desc="Evaluating"):
            labels = batch["label"].to(self.device)
            features = self._extract_features(batch)
            logits = self.classifier(features)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)

        # Overall accuracy
        top1 = (preds == labels).float().mean().item()

        # Per-class accuracy
        per_class = {}
        for i, name in enumerate(self.class_names):
            mask = labels == i
            if mask.sum() > 0:
                per_class[name] = (preds[mask] == labels[mask]).float().mean().item()
            else:
                per_class[name] = float("nan")

        return {
            "top1_accuracy": top1,
            "per_class_accuracy": per_class,
            "predictions": preds.numpy(),
            "ground_truth": labels.numpy(),
        }

    @torch.no_grad()
    def evaluate_rgb_baseline(self, test_loader) -> dict:
        """RGB baseline using the same backbone on RGB input."""
        old_use_rgb = self.use_rgb
        self.use_rgb = True
        results = self.evaluate(test_loader)
        self.use_rgb = old_use_rgb
        return results
