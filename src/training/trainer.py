"""Training loop for linear classifier on frozen Depth Anything features."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
from pathlib import Path
import os

# Disable tokenizer parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class ClassifierTrainer:
    """Simple cross-entropy classifier trainer with frozen backbone.

    Trains only the linear classifier head; backbone remains frozen.
    """

    def __init__(
        self,
        classifier: nn.Module,
        backbone: nn.Module,
        loss_fn: nn.Module,
        config,
    ):
        self.classifier = classifier
        self.backbone = backbone  # frozen
        self.loss_fn = loss_fn
        self.config = config

        self.optimizer = AdamW(
            classifier.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )

        self.device = config.device
        self.use_rgb = getattr(config, "use_rgb_input", False)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _extract_features(self, batch):
        """Extract features from depth (or RGB) images using frozen backbone."""
        if self.use_rgb:
            pil_images = batch["rgb_pil"]
        else:
            pil_images = batch["depth_pil"]

        # Convert single-channel PIL to 3-channel
        rgb_tensors = [d.convert("RGB") for d in pil_images]

        # Process with backbone's processor
        inputs = self.backbone.processor(
            images=rgb_tensors,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            features = self.backbone.encode_depth(inputs["pixel_values"])
        return features

    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.classifier.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            labels = batch["label"].to(self.device)
            features = self._extract_features(batch)
            logits = self.classifier(features)
            loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, val_loader) -> tuple:
        """Evaluate on validation set.

        Returns:
            (avg_loss, accuracy)
        """
        self.classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            labels = batch["label"].to(self.device)
            features = self._extract_features(batch)
            logits = self.classifier(features)

            loss = self.loss_fn(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def train(self, train_loader, val_loader):
        """Full training loop."""
        best_val_acc = 0.0

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            self.scheduler.step()

            print(
                f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2%}"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.classifier.state_dict(),
                           self.output_dir / "best_classifier.pt")

        # Save final model and history
        torch.save(self.classifier.state_dict(),
                   self.output_dir / "final_classifier.pt")
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"Training complete. Best Val Acc: {best_val_acc:.2%}")
        return self.history
