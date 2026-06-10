"""Synthetic depth dataset for testing the full pipeline locally."""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import List, Optional


class SyntheticDepthDataset(Dataset):
    """Generates synthetic RGB-Depth pairs with scene labels.

    Useful for testing the full pipeline without downloading real data.
    Each sample has a random RGB image (noise + simple shapes) and
    a corresponding depth map (with simple geometric patterns).
    """

    def __init__(
        self,
        num_samples: int = 200,
        image_size: int = 224,
        class_names: Optional[List[str]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed

        if class_names is None:
            self.class_names = [
                "bedroom", "living room", "bathroom", "dining room",
                "kitchen", "home office", "office", "classroom",
                "library", "bookstore", "laundry", "furniture store",
                "study",
            ]
        else:
            self.class_names = class_names

        self.rng = np.random.RandomState(seed)

        # Pre-generate all samples deterministically
        self._samples = []
        for i in range(num_samples):
            label = i % len(self.class_names)
            self._samples.append(self._generate_sample(label, i))

    def _generate_sample(self, label: int, index: int) -> dict:
        """Generate one RGB-Depth pair with deterministic pattern."""
        H, W = self.image_size, self.image_size
        local_rng = np.random.RandomState(self.seed + index * 7)

        # RGB: base color per class
        base_color = np.array([
            (label * 37) % 256,
            (label * 53) % 256,
            (label * 71) % 256,
        ], dtype=np.uint8)

        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for c in range(3):
            rgb[:, :, c] = base_color[c]

        # Add some random shapes to make images distinct
        # Circles
        for _ in range(local_rng.randint(2, 5)):
            cx = local_rng.randint(0, W)
            cy = local_rng.randint(0, H)
            r = local_rng.randint(10, 40)
            yy, xx = np.ogrid[:H, :W]
            mask = (xx - cx)**2 + (yy - cy)**2 < r**2
            rgb[mask] = (
                local_rng.randint(0, 255),
                local_rng.randint(0, 255),
                local_rng.randint(0, 255),
            )

        # Random noise
        noise = local_rng.randint(0, 30, (H, W, 3), dtype=np.uint8)
        rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Depth: create pseudo-depth with gradients + shapes
        depth = np.zeros((H, W), dtype=np.float32)

        # Base gradient (simulating perspective)
        y_grad = np.linspace(0.5, 1.0, H).reshape(H, 1)
        depth += y_grad * 5.0  # range 2.5-5.0 meters

        # Add random depth blobs
        for _ in range(local_rng.randint(2, 4)):
            cx = local_rng.randint(0, W)
            cy = local_rng.randint(0, H)
            r = local_rng.randint(15, 50)
            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            blob = np.maximum(0, 1 - dist / r) * local_rng.uniform(1.0, 3.0)
            depth += blob

        # Add noise
        depth += local_rng.randn(H, W) * 0.05
        depth = np.clip(depth, 0.1, 10.0)

        # Convert to PIL
        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)

        return {
            "rgb_pil": Image.fromarray(rgb),
            "depth_pil": Image.fromarray(depth_norm),
            "depth_raw": torch.from_numpy(depth).float(),
            "scene": self.class_names[label],
            "label": label,
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]


def create_synthetic_dataloaders(
    num_train: int = 200,
    num_val: int = 50,
    num_test: int = 50,
    batch_size: int = 16,
    image_size: int = 224,
    seed: int = 42,
):
    """Create train/val/test dataloaders with synthetic data."""
    full = SyntheticDepthDataset(
        num_samples=num_train + num_val + num_test,
        image_size=image_size,
        seed=seed,
    )

    indices = list(range(len(full)))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:num_train + num_val + num_test]

    from torch.utils.data import Subset

    def collate_fn(batch):
        return {
            "rgb_pil": [item["rgb_pil"] for item in batch],
            "depth_pil": [item["depth_pil"] for item in batch],
            "scene": [item["scene"] for item in batch],
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "depth_raw": torch.stack([item["depth_raw"] for item in batch]),
        }

    train_loader = torch.utils.data.DataLoader(
        Subset(full, train_indices), batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        Subset(full, val_indices), batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        Subset(full, test_indices), batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
