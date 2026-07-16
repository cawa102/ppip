"""Task 6 — masked-δ texture design + real-render surrogate for the monitor hijack.

The attacker controls ONLY the monitor's pixels, so an adversarial perturbation is:
  1. optimised as a masked-δ on the 224 policy-input (added pre-crop, matching the proven
     adaptive_attack path; vla_diff center-crops internally before the CE toward the teacher
     tokens), confined to the monitor mask (``apply_masked_delta`` / ``optimize_masked_delta``);
  2. projected from that image region onto the fixed monitor texture via the calibrated
     homography (``warp_pattern_to_texture``) so the render lays it back where it belongs;
  3. corrected for the render reality-gap (additive-δ proposal vs what the real renderer
     actually shows) by a surrogate measured from one real render (``calibrate_surrogate``).

Per-texture selection is stateless: candidates are scored by target-token CE / logit margin
on the post-upload render, with no env step (``select_texture``). The confinement / warp /
surrogate math is CPU-pure and unit-tested; the vla_diff optimisation and the OpenVLA
proxy scoring are GPU seams verified behind ``PPIP_GPU_TESTS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rendering.monitor import UVMap, homography_from_correspondences


def apply_masked_delta(
    frame: NDArray[np.floating[Any]],
    delta: NDArray[np.floating[Any]],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Add ``delta`` to ``frame`` ONLY where ``mask`` is True; outside is left untouched.

    The monitor threat model lets the attacker change only the pixels the monitor controls,
    so the perturbation is confined to the monitor mask. Never mutates ``frame``.
    """
    out = np.array(frame, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    out[m] = out[m] + np.asarray(delta, dtype=np.float64)[m]
    return out


def warp_pattern_to_texture(
    image_pattern: NDArray[np.uint8], uv_map: UVMap, tex_hw: tuple[int, int]
) -> NDArray[np.uint8]:
    """Inverse-warp an image-space pattern onto the fixed monitor texture canvas.

    For each texture pixel, map it into policy-image space through the texture->image
    homography (fit from the calibrated corner correspondences) and sample the pattern
    there. So when the monitor is rendered, the projected texture reproduces the pattern
    inside the monitor quad. Nearest-neighbour sampling keeps corner colours exact.
    """
    tex_h, tex_w = int(tex_hw[0]), int(tex_hw[1])
    tex_to_img = homography_from_correspondences(
        uv_map.texture_corners, uv_map.image_corners
    )
    grid_x, grid_y = np.meshgrid(np.arange(tex_w), np.arange(tex_h))
    tex_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
    img_pts = tex_to_img.apply(tex_pts)

    height, width = image_pattern.shape[:2]
    col = np.clip(np.round(img_pts[:, 0]).astype(int), 0, width - 1)
    row = np.clip(np.round(img_pts[:, 1]).astype(int), 0, height - 1)
    sampled = np.asarray(image_pattern)[row, col]
    texture: NDArray[np.uint8] = sampled.reshape(tex_h, tex_w, -1).astype(np.uint8)
    return texture


@dataclass(frozen=True)
class Surrogate:
    """The measured render reality-gap: what the real renderer shows minus the proposal.

    ``residual`` is ``real - proposed`` inside the monitor mask (zero outside), so ``apply``
    corrects any proposal in that region toward the real render; ``gap`` is the magnitude of
    the correction (max abs residual) -- the "within-the-surrogate's-gap" tolerance Task 6
    designs against before committing a texture.
    """

    residual: NDArray[np.float64]
    mask: NDArray[np.bool_]
    gap: float

    def apply(self, proposed: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Correct a proposed policy-input toward the real render (adds the residual)."""
        return np.asarray(proposed, dtype=np.float64) + self.residual


def calibrate_surrogate(
    proposed: NDArray[np.floating[Any]],
    real_render: NDArray[np.floating[Any]],
    mask: NDArray[np.bool_],
) -> Surrogate:
    """Measure the render reality-gap from one (proposed, real render) pair inside the mask.

    The optimiser reasons about an additive-δ *proposal*; the real renderer replaces monitor
    pixels with shaded/resampled content instead. This captures that discrepancy so a
    proposal can be corrected (and its residual bounded) before it is committed to a texture.
    """
    prop = np.asarray(proposed, dtype=np.float64)
    real = np.asarray(real_render, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    residual = np.zeros_like(prop)
    residual[m] = real[m] - prop[m]
    gap = float(np.abs(residual[m]).max()) if m.any() else 0.0
    return Surrogate(residual=residual, mask=m, gap=gap)


def optimize_masked_delta(
    model: Any,
    img224: NDArray[np.uint8],
    mask224: NDArray[np.bool_],
    teacher_tokens: Any,
    user_ids: Any,
    *,
    k: int = 6,
    eps: float = 0.15,
    lr: float = 2e-2,
) -> NDArray[np.float64]:  # pragma: no cover - GPU seam, verified in GPU rollout env
    """Optimise a monitor-confined additive-δ toward the teacher tokens (vla_diff CE).

    The white-box inner loop of adaptive_attack, but the perturbation is masked to the
    monitor region (``delta = eps*tanh(raw)*mask``) -- the attacker controls only the
    monitor's pixels. Deterministic (no EoT crop jitter) so the seam is reproducible;
    Task 7/8 add jitter for a replay-robust video. Returns the masked δ in ``img224``
    space, shape ``(224, 224, 3)`` float; it is a *proposal* (see calibrate_surrogate).
    """
    import torch
    import torch.nn.functional as F
    import vla_diff

    device = next(model.parameters()).device
    # Freeze the policy so autograd flows ONLY to the perturbation, not the 7B params -- as
    # adaptive_attack does. Without this, backprop allocates a gradient buffer per parameter
    # and the single-model process OOMs (~22GB) on a 24GB card.
    for param in model.parameters():
        param.requires_grad_(False)
    model.language_model.config.use_cache = False

    base = torch.from_numpy(np.asarray(img224, dtype=np.float32) / 255.0)
    base = base.permute(2, 0, 1)[None].to(device)
    mask_t = torch.from_numpy(np.asarray(mask224, dtype=np.float32))[None, None].to(device)
    teacher = teacher_tokens.view(1, vla_diff.ACTION_DIM).to(device)

    raw = torch.zeros(1, 3, 224, 224, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    for _ in range(k):
        delta = eps * torch.tanh(raw) * mask_t
        pv = vla_diff.preprocess((base + delta).clamp(0, 1))
        logits = vla_diff.action_token_logits(model, pv, user_ids, teacher)
        loss = F.cross_entropy(
            logits.reshape(vla_diff.ACTION_DIM, -1).float(),
            teacher.reshape(vla_diff.ACTION_DIM),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        delta = eps * torch.tanh(raw) * mask_t
    out: NDArray[np.float64] = delta[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float64)
    return out


def _target_token_ce(
    model: Any, img224: NDArray[np.uint8], user_ids: Any, teacher: Any
) -> float:  # pragma: no cover - GPU seam, verified in GPU rollout env
    """Teacher-token cross-entropy of the policy on a 224 policy-input (no env step)."""
    import torch
    import torch.nn.functional as F
    import vla_diff

    device = next(model.parameters()).device
    img = torch.from_numpy(np.asarray(img224, dtype=np.float32) / 255.0)
    img = img.permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        pv = vla_diff.preprocess(img)
        logits = vla_diff.action_token_logits(model, pv, user_ids, teacher.view(1, -1))
        ce = F.cross_entropy(
            logits.reshape(vla_diff.ACTION_DIM, -1).float(),
            teacher.view(vla_diff.ACTION_DIM),
        )
    return float(ce.item())


def select_texture(
    candidate_textures: list[NDArray[np.uint8]],
    backend: Any,
    env: Any,
    handle: Any,
    teacher_tokens: Any,
    user_ids: Any,
    *,
    resize_size: int = 224,
) -> tuple[NDArray[np.uint8], list[float]]:  # pragma: no cover - GPU seam, verified in GPU env
    """Pick the candidate texture with the lowest teacher-token CE on its real render.

    STATELESS proxy scoring: each candidate is uploaded and rendered fresh (the Task-4
    post-upload render), scored by target-token CE, and NO ``env.step`` is ever taken -- the
    committed rollout is never advanced to test a candidate. Leaves the monitor showing the
    winner (which the next real step re-uploads anyway). Returns ``(best_texture, scores)``.
    """
    from rendering.monitor import _policy_input_frame

    if backend._policy is None:
        backend._policy = backend._load_policy()
    model = backend._policy[0]

    scores: list[float] = []
    for texture in candidate_textures:
        handle.upload(np.ascontiguousarray(np.asarray(texture, dtype=np.uint8)))
        env.sim.forward()
        frame = _policy_input_frame(env, resize_size)
        scores.append(_target_token_ce(model, frame, user_ids, teacher_tokens))

    best = int(np.argmin(scores))
    best_texture = np.ascontiguousarray(np.asarray(candidate_textures[best], dtype=np.uint8))
    handle.upload(best_texture)
    env.sim.forward()
    return best_texture, scores
