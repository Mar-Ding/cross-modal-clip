"""Depth Anything V2 wrapper for feature extraction.

Uses the DINOv2 backbone from Depth Anything V2, with optional loading
of depth-pretrained weights. Falls back to plain DINOv2 if the Depth
Anything .pth file is not available.
"""

import torch
import torch.nn as nn
from transformers import Dinov2Model, AutoImageProcessor
from typing import Optional
import os


def _load_depth_anything_weights(model: nn.Module, pth_path: str, device: str) -> bool:
    """Load Depth Anything V2 pretrained weights into a Dinov2Model.

    Depth Anything V2 weights use 'pretrained.' prefix and combined QKV
    format. We need to map them to HuggingFace DINOv2's separate Q, K, V
    and matching key names.

    Args:
        model: Dinov2Model instance (weights already loaded from HF)
        pth_path: Path to depth_anything_v2_vit*.pth
        device: Target device

    Returns:
        True if weights were loaded successfully
    """
    if not os.path.exists(pth_path):
        print(f"  Depth weights not found at {pth_path}")
        return False

    try:
        state_dict = torch.load(pth_path, map_location=device)

        # Build mapping: DA key -> HF DINOv2 key
        da_to_hf = {}

        # 1) Embeddings
        emb_map = {
            "pretrained.cls_token": "embeddings.cls_token",
            "pretrained.mask_token": "embeddings.mask_token",
            "pretrained.pos_embed": "embeddings.position_embeddings",
            "pretrained.patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
            "pretrained.patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
        }
        # 2) Final norm
        norm_map = {
            "pretrained.norm.weight": "layernorm.weight",
            "pretrained.norm.bias": "layernorm.bias",
        }

        # Direct mappings (QKV excluded, handled separately)
        direct_map = {
            **emb_map,
            **norm_map,
        }

        for da_key, hf_key in direct_map.items():
            if da_key in state_dict:
                da_to_hf[da_key] = hf_key

        # 3) Per-block mappings (for 24 blocks in ViT-B)
        num_blocks = 12  # Dinovov2Base has 12 blocks
        # Let's detect from model
        block_keys = [k for k in model.state_dict().keys() if k.startswith("encoder.layer.")]
        block_ids = set()
        for k in block_keys:
            parts = k.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                block_ids.add(int(parts[2]))
        num_blocks = max(block_ids) + 1 if block_ids else 12

        for blk in range(num_blocks):
            prefix_da = f"pretrained.blocks.{blk}."
            prefix_hf = f"encoder.layer.{blk}."

            # Norms (direct)
            da_to_hf[f"{prefix_da}norm1.weight"] = f"{prefix_hf}norm1.weight"
            da_to_hf[f"{prefix_da}norm1.bias"] = f"{prefix_hf}norm1.bias"
            da_to_hf[f"{prefix_da}norm2.weight"] = f"{prefix_hf}norm2.weight"
            da_to_hf[f"{prefix_da}norm2.bias"] = f"{prefix_hf}norm2.bias"

            # Layer scales
            da_to_hf[f"{prefix_da}ls1.gamma"] = f"{prefix_hf}layer_scale1.lambda1"
            da_to_hf[f"{prefix_da}ls2.gamma"] = f"{prefix_hf}layer_scale2.lambda1"

            # MLP
            da_to_hf[f"{prefix_da}mlp.fc1.weight"] = f"{prefix_hf}mlp.fc1.weight"
            da_to_hf[f"{prefix_da}mlp.fc1.bias"] = f"{prefix_hf}mlp.fc1.bias"
            da_to_hf[f"{prefix_da}mlp.fc2.weight"] = f"{prefix_hf}mlp.fc2.weight"
            da_to_hf[f"{prefix_da}mlp.fc2.bias"] = f"{prefix_hf}mlp.fc2.bias"

            # Attention projection output (direct)
            da_to_hf[f"{prefix_da}attn.proj.weight"] = f"{prefix_hf}attention.output.dense.weight"
            da_to_hf[f"{prefix_da}attn.proj.bias"] = f"{prefix_hf}attention.output.dense.bias"

            # QKV is combined in DA, separate in HF
            qkv_key = f"{prefix_da}attn.qkv.weight"
            qkv_bias_key = f"{prefix_da}attn.qkv.bias"
            if qkv_key in state_dict:
                qkv_w = state_dict[qkv_key]  # (3*D, D)
                qkv_b = state_dict[qkv_bias_key]  # (3*D,)
                D = qkv_w.shape[1]
                q_w, k_w, v_w = qkv_w.chunk(3, dim=0)
                q_b, k_b, v_b = qkv_b.chunk(3, dim=0)

                q_hf = f"{prefix_hf}attention.attention.query"
                k_hf = f"{prefix_hf}attention.attention.key"
                v_hf = f"{prefix_hf}attention.attention.value"

                da_to_hf[qkv_key] = (q_hf + ".weight", q_w)
                da_to_hf[qkv_bias_key + ".q"] = (q_hf + ".bias", q_b)
                da_to_hf[qkv_bias_key + ".k"] = (k_hf + ".weight", k_w)
                da_to_hf[qkv_bias_key + ".kb"] = (k_hf + ".bias", k_b)
                da_to_hf[qkv_bias_key + ".v"] = (v_hf + ".weight", v_w)
                da_to_hf[qkv_bias_key + ".vb"] = (v_hf + ".bias", v_b)
                continue  # Don't add to direct da_to_hf

        # Build the HF-compatible state dict
        hf_state = {}
        loaded_count = 0
        for da_key, mapping in da_to_hf.items():
            if da_key not in state_dict:
                continue
            if isinstance(mapping, str):
                # Direct mapping
                hf_state[mapping] = state_dict[da_key]
                loaded_count += 1
            elif isinstance(mapping, tuple):
                # Already processed (split QKV)
                hf_state[mapping[0]] = mapping[1]
                loaded_count += 1

        # Load into model (non-strict to skip head keys)
        missing, unexpected = model.load_state_dict(hf_state, strict=False)
        print(f"  Depth Anything weights: {loaded_count} keys mapped")
        if missing:
            print(f"    Missing (expected): {len(missing)} keys (non-backbone)")
        if unexpected:
            print(f"    Unexpected: {len(unexpected)} keys")
        return True

    except Exception as e:
        print(f"  Failed to load Depth Anything weights: {e}")
        import traceback
        traceback.print_exc()
        return False


class DepthAnythingWrapper(nn.Module):
    """Frozen Depth Anything V2 encoder for depth feature extraction.

    Depth Anything V2 uses a DINOv2 backbone internally. We extract
    the [CLS] token as the global depth representation.

    Supports loading depth-pretrained weights from a .pth file,
    automatically handling the QKV split and key mapping.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "cpu",
        depth_weights_path: Optional[str] = None,
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name

        # Determine feature dim from model name
        if "small" in model_name:
            self.feature_dim = 384
        elif "base" in model_name:
            self.feature_dim = 768
        elif "large" in model_name:
            self.feature_dim = 1024
        elif "giant" in model_name or "gaint" in model_name:
            self.feature_dim = 1536
        else:
            self.feature_dim = 768

        # Load the DINOv2 model (via HF mirror)
        self.model = Dinov2Model.from_pretrained(model_name).to(device)

        # Load Depth Anything V2 pretrained weights (if available)
        if depth_weights_path:
            loaded = _load_depth_anything_weights(self.model, depth_weights_path, device)
            if loaded:
                print("  Using Depth Anything V2 pretrained backbone!")
            else:
                print("  Falling back to plain DINOv2 backbone")
        else:
            print(f"  Using plain DINOv2 ({model_name})")

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Image processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_depth(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract global depth features using [CLS] token.

        Args:
            pixel_values: (B, 3, H, W) normalized depth image.
                          Depth is single-channel, repeated to 3 channels.

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
        """Get all patch tokens (for cross-attention adapter if needed)."""
        outputs = self.model(pixel_values)
        return outputs.last_hidden_state  # (B, N+1, D)
