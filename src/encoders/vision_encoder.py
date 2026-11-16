"""
Frozen Pretrained Vision Feature Extractor and Letterbox Geometry Utility.

Implements geometry-preserving white-canvas letterboxing onto 224x224, exact 7x7
spatial patch validity mask computation (M_patch_q in {0, -1e9}^49), and frozen
CLIP ViT-B/16 feature extraction (14x14 -> 2x2 average pooled 7x7 spatial patches
for ACD-MMKTv3 and masked/CLS pooled vectors for MG-MMKT/v2).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModel


def letterbox_image_and_mask(
    img_input: Union[str, Image.Image],
    target_size: int = 224,
    grid_size: int = 7,
) -> Tuple[Image.Image, torch.Tensor, torch.Tensor, Tuple[int, int, int, int]]:
    """
    Resize an image while preserving aspect ratio, center on a pure white 224x224
    canvas, and compute the exact 7x7 spatial patch validity mask.

    Args:
        img_input: File path or PIL Image.
        target_size: Square canvas edge length in pixels (default: 224).
        grid_size: Spatial patch grid edge count (default: 7 -> 49 patches).

    Returns:
        Tuple of:
        - letterboxed_img: 224x224 RGB PIL Image on pure white (255, 255, 255) background.
        - M_patch: [49] float32 additive attention mask (0.0 valid, -1e9 padded).
        - patch_valid_mask: [49] bool mask (True if overlapping actual content).
        - letterbox_bbox: (x0, y0, x1, y1) bounding box of actual image content on 224x224.
    """
    if isinstance(img_input, str):
        img = Image.open(img_input)
    else:
        img = img_input

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        canvas = Image.new("RGB", img.size, (255, 255, 255))
        canvas.paste(img, mask=img.split()[3])
        img = canvas
    else:
        img = img.convert("RGB")

    w_orig, h_orig = img.size
    scale = target_size / float(max(1, max(w_orig, h_orig)))
    w_new = max(1, int(round(w_orig * scale)))
    h_new = max(1, int(round(h_orig * scale)))

    resized = img.resize((w_new, h_new), Image.BICUBIC)
    letterboxed = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    x0 = (target_size - w_new) // 2
    y0 = (target_size - h_new) // 2
    x1 = x0 + w_new
    y1 = y0 + h_new
    letterboxed.paste(resized, (x0, y0))

    # Row-major 7x7 spatial patch validity grid matching 14x14 -> 2x2 adaptive pooling
    patch_dim = target_size // grid_size
    grid_mask = np.zeros((grid_size, grid_size), dtype=bool)
    for r in range(grid_size):
        cell_y0 = r * patch_dim
        cell_y1 = cell_y0 + patch_dim
        for c in range(grid_size):
            cell_x0 = c * patch_dim
            cell_x1 = cell_x0 + patch_dim
            overlap_w = max(0, min(cell_x1, x1) - max(cell_x0, x0))
            overlap_h = max(0, min(cell_y1, y1) - max(cell_y0, y0))
            if overlap_w > 0 and overlap_h > 0:
                grid_mask[r, c] = True

    # Safety guarantee: at least the center patch stays open to prevent Softmax NaN
    if not grid_mask.any():
        grid_mask[grid_size // 2, grid_size // 2] = True

    patch_valid_mask = torch.from_numpy(grid_mask.reshape(-1)).bool()
    m_patch = torch.zeros(grid_size * grid_size, dtype=torch.float32)
    m_patch[~patch_valid_mask] = -1e9

    return letterboxed, m_patch, patch_valid_mask, (x0, y0, x1, y1)


class FrozenVisionEncoder:
    """
    Frozen CLIP ViT-B/16 encoder producing 7x7 spatial patch matrices and global vectors.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        device: Optional[torch.device] = None,
        local_files_only: bool = True,
    ):
        """
        Initialize frozen CLIP Vision backbone and image processor.

        Args:
            model_name: Pretrained CLIP Vision model identifier or local path.
            device: Compute device; auto-selected if None.
            local_files_only: Prefer local HF cache before remote download.
        """
        self.model_name = model_name
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        try:
            self.processor = CLIPImageProcessor.from_pretrained(
                model_name, local_files_only=local_files_only
            )
            self.model = CLIPVisionModel.from_pretrained(
                model_name, local_files_only=local_files_only
            )
        except Exception:
            self.processor = CLIPImageProcessor.from_pretrained(model_name)
            self.model = CLIPVisionModel.from_pretrained(model_name)

        self.model = self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode_images(
        self,
        images: List[Image.Image],
        patch_valid_masks: Optional[torch.Tensor] = None,
        batch_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract 7x7 spatial patch matrices and global pooled vectors from 224x224 RGB images.

        Args:
            images: List of N PIL RGB images (already letterboxed to 224x224).
            patch_valid_masks: Optional [N, 49] bool masks for masked spatial pooling.
            batch_size: Mini-batch size.

        Returns:
            Tuple of:
            - patch_features: [N, 49, 768] float16 spatial grid features (row-major 7x7).
            - cls_features: [N, 768] float16 global [CLS] representations.
            - masked_patch_pooled: [N, 768] float16 mean across valid 7x7 spatial patches.
        """
        num_imgs = len(images)
        hidden_dim = int(self.model.config.hidden_size)
        patch_features = torch.zeros((num_imgs, 49, hidden_dim), dtype=torch.float16)
        cls_features = torch.zeros((num_imgs, hidden_dim), dtype=torch.float16)
        masked_patch_pooled = torch.zeros((num_imgs, hidden_dim), dtype=torch.float16)

        with torch.no_grad():
            for start in range(0, num_imgs, batch_size):
                batch_imgs = images[start : start + batch_size]
                pixel_inputs = self.processor(images=batch_imgs, return_tensors="pt")
                pixel_values = pixel_inputs["pixel_values"].to(self.device)

                out = self.model(pixel_values=pixel_values)
                last_hidden = out.last_hidden_state  # [B, 197, 768]
                cls_tokens = last_hidden[:, 0, :]    # [B, 768]
                raw_patches = last_hidden[:, 1:, :]  # [B, 196, 768]

                b_dim = raw_patches.size(0)
                # Row-major 14x14 -> 2x2 spatial pooling -> row-major 7x7 (idx = 7*r + c)
                p_grid = raw_patches.view(b_dim, 14, 14, hidden_dim).permute(0, 3, 1, 2)
                p_pooled = (
                    F.adaptive_avg_pool2d(p_grid, (7, 7))
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    .view(b_dim, 49, hidden_dim)
                )

                end = start + b_dim
                patch_features[start:end] = p_pooled.cpu().to(torch.float16)
                cls_features[start:end] = cls_tokens.cpu().to(torch.float16)

                if patch_valid_masks is not None:
                    sub_mask = patch_valid_masks[start:end].unsqueeze(-1).float().to(self.device)
                    p_sum = (p_pooled.float() * sub_mask).sum(dim=1)
                    p_cnt = sub_mask.sum(dim=1).clamp(min=1e-8)
                    masked_patch_pooled[start:end] = (p_sum / p_cnt).cpu().to(torch.float16)
                else:
                    masked_patch_pooled[start:end] = p_pooled.mean(dim=1).cpu().to(torch.float16)

        return patch_features, cls_features, masked_patch_pooled

    def release(self) -> None:
        """Free model weights from device memory after offline extraction."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
