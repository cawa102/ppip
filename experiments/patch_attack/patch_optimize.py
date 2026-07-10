"""Stage 2 — optimize a digital camera-space patch (behavioral distillation).

Push OpenVLA(frame ⊕ patch, USER=alphabet_soup) toward the teacher's action tokens
OpenVLA(frame, TARGET=salad_dressing), over the collected states, with EoT. Loss = CE on the
7 action-token logits. Only the patch pixels have gradients; the 7B model is frozen.
"""

from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HOME = os.path.expanduser("~")
for p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, p))

import patch_config as C  # noqa: E402
import vla_diff  # noqa: E402
from evaluator.openvla_backend import OpenVLARolloutBackend  # noqa: E402

DEVICE = "cuda"
ITERS = int(os.environ.get("PATCH_ATTACK_ITERS", "800"))
BATCH = int(os.environ.get("PATCH_ATTACK_BATCH", "6"))
LR = float(os.environ.get("PATCH_ATTACK_LR", "1e-3"))  # 1e-2 diverged; 1e-3 descends (verified)
LOG_EVERY = int(os.environ.get("PATCH_ATTACK_LOG_EVERY", "25"))


def main() -> None:
    states_npz = os.environ.get("PATCH_ATTACK_STATES_NPZ", C.STATES_NPZ)
    data = np.load(states_npz)
    images = torch.from_numpy(data["images"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    tokens = torch.from_numpy(data["tokens"].astype(np.int64))
    n = images.shape[0]
    print(
        f"[opt] {n} states from {states_npz}; token range [{int(tokens.min())},{int(tokens.max())}]",
        flush=True,
    )

    backend = OpenVLARolloutBackend()
    model, processor, _, _ = backend._load_policy()
    model.eval()
    for pm in model.parameters():
        pm.requires_grad_(False)
    # NOTE: gradient checkpointing in eval() mode produced WRONG gradients here (loss diverged);
    # a minimal no-checkpoint test descended cleanly. Keep it OFF by default (batch stays small).
    if os.environ.get("PATCH_ATTACK_GRADCKPT", "0") == "1":
        model.language_model.gradient_checkpointing_enable()
    model.language_model.config.use_cache = False

    prompt = f"In: What action should the robot take to {C.USER_TASK.lower()}?\nOut:"
    ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    r0, c0, ph, pw = C.PATCH_ROW0, C.PATCH_COL0, C.PATCH_H, C.PATCH_W
    clip = float(os.environ.get("PATCH_ATTACK_CLIP", "5e-3"))
    ckpt_every = int(os.environ.get("PATCH_ATTACK_CKPT_EVERY", "20"))
    eot = os.environ.get("PATCH_ATTACK_EOT", "1") == "1"
    ckpt_path = os.path.join(C.RUN_DIR, "raw_ckpt.pt")

    # Resume from checkpoint if present -> resilient to the env's intermittent process kills.
    raw = torch.zeros(3, ph, pw, device=DEVICE)
    start_it = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE)
        raw = ck["raw"].to(DEVICE)
        start_it = int(ck["it"])
        print(f"[opt] resumed from it={start_it}", flush=True)
    raw.requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=LR)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE)
        if "opt" in ck:
            with contextlib.suppress(Exception):
                opt.load_state_dict(ck["opt"])

    def _save(it: int) -> None:
        patch_np = torch.sigmoid(raw).detach().cpu().permute(1, 2, 0).numpy()
        torch.save({"raw": raw.detach().cpu(), "opt": opt.state_dict(), "it": it}, ckpt_path)
        np.save(C.PATCH_NPY, patch_np)
        from PIL import Image
        Image.fromarray((patch_np * 255).astype(np.uint8)).save(
            os.path.join(C.RUN_DIR, "patch_v1_preview.png")
        )

    gen = torch.Generator(device="cpu").manual_seed(1234 + start_it)
    for it in range(start_it, ITERS):
        idx = torch.randint(0, n, (BATCH,), generator=gen)
        img = images[idx].to(DEVICE)  # [B,3,224,224]
        tgt = tokens[idx].to(DEVICE)  # [B,7]
        patch = torch.sigmoid(raw)  # [3,ph,pw] in (0,1)
        if eot:  # light EoT: brightness + small placement jitter
            bright = 0.9 + 0.2 * torch.rand(1, device=DEVICE)
            dr = int(torch.randint(-3, 4, (1,), generator=gen).item())
            dc = int(torch.randint(-3, 4, (1,), generator=gen).item())
            patch = (patch * bright).clamp(0, 1)
        else:
            dr = dc = 0
        rr, cc = r0 + dr, c0 + dc
        patched = img.clone()
        patched[:, :, rr : rr + ph, cc : cc + pw] = patch
        pv = vla_diff.preprocess(patched)
        logits = vla_diff.action_token_logits(model, pv, ids.repeat(BATCH, 1), tgt)  # [B,7,V]
        loss = F.cross_entropy(logits.reshape(BATCH * 7, -1).float(), tgt.reshape(BATCH * 7))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], clip)  # guard vs noisy bf16 grads -> divergence
        opt.step()
        if it % LOG_EVERY == 0 or it == ITERS - 1:
            with torch.no_grad():
                acc = (logits.argmax(-1) == tgt).float().mean().item()
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"[opt] it={it:4d} loss={loss.item():.4f} tok_acc={acc:.3f} "
                  f"maxmem={mem:.1f}GB", flush=True)
        if it % ckpt_every == 0 or it == ITERS - 1:
            _save(it + 1)

    _save(ITERS)
    print(f"[opt] DONE it={ITERS}; saved patch -> {C.PATCH_NPY} (region r{r0} c{c0} {ph}x{pw})",
          flush=True)


if __name__ == "__main__":
    main()
