"""Full-image L-inf adversarial perturbation (Tier-A vision-layer hijack).

Optimize a universal-over-seed-0-trajectory perturbation delta = eps*tanh(raw) so that
OpenVLA(clip(frame+delta), USER=alphabet_soup) emits the teacher action tokens
OpenVLA(frame, TARGET=salad_dressing). Much higher capacity than a localized patch.
CE on the 7 action tokens, gradient accumulation for a stable gradient, checkpoint/resume
(the env kills multi-minute GPU processes ~every few min). delta.npy is the artifact.
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
ITERS = int(os.environ.get("DELTA_ITERS", "150"))
MB = int(os.environ.get("DELTA_MB", "3"))          # microbatch (memory driver)
ACCUM = int(os.environ.get("DELTA_ACCUM", "5"))    # microbatches accumulated per step
LR = float(os.environ.get("DELTA_LR", "5e-3"))
EPS = float(os.environ.get("DELTA_EPS", "0.125"))  # L-inf bound (~32/255)
CKPT_EVERY = int(os.environ.get("DELTA_CKPT_EVERY", "4"))
# Per-action-dim CE weights: (dx,dy,dz, dR,dP,dY, grip). Emphasize position -> object redirection.
POS_W = [float(x) for x in os.environ.get("DELTA_POS_W", "3,3,3,0.3,0.3,0.3,1").split(",")]
DELTA_NPY = os.path.join(C.RUN_DIR, "delta.npy")
CKPT = os.path.join(C.RUN_DIR, "delta_ckpt.pt")


def main() -> None:
    data = np.load(os.environ.get("DELTA_STATES_NPZ", C.STATES_NPZ))
    images = torch.from_numpy(data["images"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    tokens = torch.from_numpy(data["tokens"].astype(np.int64))
    n = images.shape[0]
    print(f"[delta] {n} seed-0 states; eps={EPS:.3f} lr={LR} MB={MB} ACCUM={ACCUM}", flush=True)

    backend = OpenVLARolloutBackend()
    model, processor, _, _ = backend._load_policy()
    model.eval()
    for pm in model.parameters():
        pm.requires_grad_(False)
    model.language_model.config.use_cache = False

    prompt = f"In: What action should the robot take to {C.USER_TASK.lower()}?\nOut:"
    ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    raw = torch.zeros(3, 224, 224, device=DEVICE)
    start_it = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=DEVICE)
        raw = ck["raw"].to(DEVICE)
        start_it = int(ck["it"])
        print(f"[delta] resumed it={start_it}", flush=True)
    raw.requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=LR)
    if os.path.exists(CKPT):
        with contextlib.suppress(Exception):
            opt.load_state_dict(torch.load(CKPT, map_location=DEVICE)["opt"])

    def delta_of(r: torch.Tensor) -> torch.Tensor:
        return EPS * torch.tanh(r)

    def _save(it: int) -> None:
        # Atomic: write to tmp then rename, so a mid-write kill can't corrupt the checkpoint.
        torch.save({"raw": raw.detach().cpu(), "opt": opt.state_dict(), "it": it}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)
        np.save(DELTA_NPY + ".tmp.npy", delta_of(raw).detach().cpu().permute(1, 2, 0).numpy())
        os.replace(DELTA_NPY + ".tmp.npy", DELTA_NPY)

    def full_acc() -> float:
        """Clean tok-match over ALL states (the real convergence signal)."""
        d = delta_of(raw)
        hits = tot = 0
        with torch.no_grad():
            for i in range(0, n, MB):
                im = images[i : i + MB].to(DEVICE)
                tg = tokens[i : i + MB].to(DEVICE)
                pv = vla_diff.preprocess((im + d).clamp(0, 1))
                lg = vla_diff.action_token_logits(model, pv, ids.repeat(im.shape[0], 1), tg)
                hits += int((lg.argmax(-1) == tg).sum())
                tot += tg.numel()
        return hits / tot

    run_iters = int(os.environ.get("DELTA_RUN_ITERS", "10000"))  # cap THIS invocation (kill-avoid)
    end_it = min(ITERS, start_it + run_iters)
    gen = torch.Generator(device="cpu").manual_seed(7 + start_it)
    for it in range(start_it, end_it):
        opt.zero_grad()
        tot_loss = 0.0
        for _ in range(ACCUM):
            idx = torch.randint(0, n, (MB,), generator=gen)
            im = images[idx].to(DEVICE)
            tg = tokens[idx].to(DEVICE)
            d = delta_of(raw)  # fresh graph each microbatch -> frees after backward
            pv = vla_diff.preprocess((im + d).clamp(0, 1))
            logits = vla_diff.action_token_logits(model, pv, ids.repeat(MB, 1), tg)
            # Weight per action dim: (dx,dy,dz) drive WHICH object the arm approaches -> redirection;
            # rotation/gripper matter far less for object-selection. Focus the attack on motion.
            ce = F.cross_entropy(
                logits.reshape(MB * 7, -1).float(), tg.reshape(MB * 7), reduction="none"
            ).reshape(MB, 7)
            w = torch.tensor(POS_W, device=DEVICE).view(1, 7)
            loss = (ce * w).sum(dim=1).mean() / (w.sum() / 7) / ACCUM
            loss.backward()
            tot_loss += float(loss.item())
        torch.nn.utils.clip_grad_norm_([raw], float(os.environ.get("DELTA_CLIP", "1.0")))
        opt.step()
        if it == end_it - 1 and os.environ.get("DELTA_FULLACC", "0") == "1":
            acc = full_acc()
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"[delta] it={it:4d} loss={tot_loss:.4f} full_tok_acc={acc:.3f} "
                  f"maxmem={mem:.1f}GB", flush=True)
        elif it % 5 == 0:
            print(f"[delta] it={it:4d} loss={tot_loss:.4f} "
                  f"linf={EPS * float(torch.tanh(raw).abs().max()):.3f}", flush=True)
        if it % CKPT_EVERY == 0 or it == ITERS - 1:
            _save(it + 1)
    _save(end_it)
    if end_it >= ITERS:
        print(f"[delta] DONE it={ITERS}; saved -> {DELTA_NPY}", flush=True)
    else:
        print(f"[delta] PAUSED at it={end_it}/{ITERS}; saved -> {DELTA_NPY}", flush=True)


if __name__ == "__main__":
    main()
