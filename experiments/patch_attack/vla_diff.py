"""Differentiable OpenVLA image→action-token-logits path for white-box patch attacks.

Replicates the exact inference preprocessing (get_vla_action's 0.9 center-crop + the
PrismaticImageProcessor's dual normalization) in torch so gradients flow from an input
image tensor to the 7 action-token logits. Everything here is autograd-friendly.

Verified constants (from processor.image_processor):
  SigLIP  norm: mean [0.484375, 0.455078125, 0.40625], std [0.228515625, 0.2236328125, 0.224609375]
  DINOv2  norm: mean/std 0.5
  center-crop area 0.9 -> side fraction sqrt(0.9); tf.image.crop_and_resize == bilinear, align_corners.
Action tokens: id = 32000 - bin, bin in [1,256] -> ids in [31744, 31999]. Token 29871 (leading
space) is appended before the action block by predict_action and MUST be present.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_MEAN0 = (0.484375, 0.455078125, 0.40625)
_STD0 = (0.228515625, 0.2236328125, 0.224609375)
_MEAN1 = (0.5, 0.5, 0.5)
_STD1 = (0.5, 0.5, 0.5)
_CROP_SIDE = 0.9**0.5  # area 0.9 -> side sqrt(0.9)
_SPACE_TOKEN = 29871
ACTION_DIM = 7


def center_crop_resize(img: torch.Tensor, side: float = _CROP_SIDE) -> torch.Tensor:
    """Differentiable center-crop (area=side^2) + bilinear resize to 224, matching TF.

    img: [B,3,224,224] float in [0,1]. Uses grid_sample(align_corners=True) over the central
    fractional box [ (1-side)/2 , (1+side)/2 ] to mirror tf.image.crop_and_resize.
    """
    b, _, h, w = img.shape
    lin = torch.linspace(0.0, 1.0, 224, device=img.device, dtype=img.dtype)
    coords = (1.0 - side) / 2.0 + lin * side  # normalized [0,1] input positions
    g = 2.0 * coords - 1.0  # to [-1,1]
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    return F.grid_sample(img, grid, mode="bilinear", align_corners=True)


def preprocess(img224: torch.Tensor, do_crop: bool = True, side: float = _CROP_SIDE) -> torch.Tensor:
    """[B,3,224,224] float[0,1] -> [B,6,224,224] pixel_values (SigLIP norm ⊕ DINOv2 norm)."""
    x = center_crop_resize(img224, side) if do_crop else img224

    def _norm(t: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
        m = torch.tensor(mean, device=t.device, dtype=t.dtype).view(1, 3, 1, 1)
        s = torch.tensor(std, device=t.device, dtype=t.dtype).view(1, 3, 1, 1)
        return (t - m) / s

    return torch.cat([_norm(x, _MEAN0, _STD0), _norm(x, _MEAN1, _STD1)], dim=1)


def build_multimodal_embeds(model, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Reproduce PrismaticForConditionalGeneration's [BOS | 256 patches | text] embedding cat."""
    patch_features = model.vision_backbone(pixel_values.to(model.dtype))
    projected = model.projector(patch_features)  # [B,256,4096]
    text_embeds = model.get_input_embeddings()(input_ids)  # [B,T,4096]
    return torch.cat(
        [text_embeds[:, :1, :], projected, text_embeds[:, 1:, :]], dim=1
    )


def action_token_logits(
    model,
    pixel_values: torch.Tensor,
    prompt_ids: torch.Tensor,
    target_action_ids: torch.Tensor,
) -> torch.Tensor:
    """Teacher-forced logits at the 7 action positions.

    prompt_ids: [1,T] the text prompt token ids ending in ...':' (space token appended here).
    target_action_ids: [1,7] the action token ids to teacher-force.
    Returns logits [1,7,vocab] predicting each of the 7 action tokens.
    """
    dev = pixel_values.device
    # Append the SentencePiece leading-space token exactly like predict_action (batch-safe).
    if prompt_ids[0, -1].item() != _SPACE_TOKEN:
        space = torch.full((prompt_ids.shape[0], 1), _SPACE_TOKEN, device=dev, dtype=prompt_ids.dtype)
        prompt_ids = torch.cat([prompt_ids, space], dim=1)

    mm = build_multimodal_embeds(model, pixel_values, prompt_ids)  # [1, M, 4096]
    action_embeds = model.get_input_embeddings()(target_action_ids)  # [1,7,4096]
    full = torch.cat([mm, action_embeds], dim=1)  # [1, M+7, 4096]

    out = model.language_model(inputs_embeds=full, use_cache=False)
    logits = out.logits  # [1, M+7, vocab]
    m = mm.shape[1]
    # token at seq index m+k (k=0..6) is predicted by logits at index m-1+k
    idx = torch.arange(m - 1, m - 1 + ACTION_DIM, device=dev)
    return logits[:, idx, :]  # [1,7,vocab]
