"""CLIP model wrapper for feature extraction."""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from typing import Union, List


class CLIPWrapper(nn.Module):
    """Frozen CLIP model for extracting vision and text features."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Freeze all CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        # Store feature dim (projection dim = 512 for ViT-B/32)
        self.vision_dim = self.model.config.projection_dim
        self.text_dim = self.model.config.projection_dim
        # Also store raw vision model dim (768 for ViT-B/32)
        self.vision_hidden_dim = self.model.config.vision_config.hidden_size

    @torch.no_grad()
    def encode_rgb(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode RGB images into CLIP vision embeddings."""
        output = self.model.get_image_features(pixel_values)
        # Handle transformers 5.x returning output objects
        if hasattr(output, "pooler_output"):
            features = output.pooler_output
        else:
            features = output
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    @torch.no_grad()
    def encode_rgb_patch_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get patch token embeddings from CLIP vision encoder.

        Returns:
            (B, N+1, D) including [CLS] token at index 0
        """
        vision_model = self.model.vision_model
        outputs = vision_model(pixel_values, output_hidden_states=False)
        return outputs.last_hidden_state

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text prompts into CLIP text embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)

        output = self.model.get_text_features(**inputs)
        if hasattr(output, "pooler_output"):
            features = output.pooler_output
        else:
            features = output
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def get_class_text_embeds(self, class_names: List[str],
                              prefix: str = "a photo of a {}") -> torch.Tensor:
        """Get text embeddings for zero-shot class names."""
        prompts = [prefix.format(name) for name in class_names]
        return self.encode_text(prompts)

    @property
    def feature_dim(self) -> int:
        return self.vision_dim
