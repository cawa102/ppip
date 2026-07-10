"""Correctness gate for the differentiable OpenVLA path (S1 + S2).

(A) my preprocess(raw224) ≈ the real pixel_values the model consumes (via get_vla_action).
(B) my teacher-forced 7-token forward reproduces the model's greedy action tokens EXACTLY.

If both pass, gradients from an input image to the action-token logits are trustworthy and
the patch optimizer can be built on top.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch", "src"))
sys.path.insert(0, os.path.join(HOME, "autoresearch"))
sys.path.insert(0, os.path.join(HOME, "openvla"))
sys.path.insert(0, os.path.join(HOME, "autoresearch", "experiments", "patch_attack"))

import vla_diff  # noqa: E402
from evaluator.openvla_backend import OpenVLARolloutBackend  # noqa: E402

IMG = os.path.join(
    HOME,
    "autoresearch/runs/autoresearch-goal/candidates/"
    "g_r2b_base_alphabet_soup/seed0_ep0_first.png",
)
TASK = "pick up the alphabet soup and place it in the basket"
DEVICE = "cuda"


def main() -> None:
    backend = OpenVLARolloutBackend()
    model, processor, cfg, resize_size = backend._load_policy()
    model.eval()

    # --- reference path: exactly mirror get_vla_action + predict_action ---
    import tensorflow as tf
    from experiments.robot.openvla_utils import crop_and_resize

    pil = Image.open(IMG).convert("RGB")
    im = tf.convert_to_tensor(np.array(pil))
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, tf.uint8, saturate=True)
    cropped_pil = Image.fromarray(im.numpy())
    prompt = f"In: What action should the robot take to {TASK.lower()}?\nOut:"
    inputs = processor(prompt, cropped_pil).to(DEVICE, dtype=torch.bfloat16)

    captured: dict = {}

    def cap(m, args, kwargs):
        pv = kwargs.get("pixel_values")
        if pv is None and len(args) >= 3:
            pv = args[2]
        if pv is not None and "pv" not in captured:
            captured["pv"] = pv.detach().float().cpu()

    h = model.register_forward_pre_hook(cap, with_kwargs=True)

    input_ids = inputs["input_ids"]
    if input_ids[0, -1].item() != vla_diff._SPACE_TOKEN:
        sp = torch.tensor([[vla_diff._SPACE_TOKEN]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, sp], dim=1)
    with torch.no_grad():
        gen = model.generate(
            input_ids, pixel_values=inputs["pixel_values"], max_new_tokens=7, do_sample=False
        )
    ref_tokens = gen[0, -7:].detach().cpu()
    h.remove()
    real_pv = captured["pv"]  # [1,6,224,224]

    # --- my differentiable path from the RAW 224 image ---
    raw = np.array(Image.open(IMG).convert("RGB"), dtype=np.float32) / 255.0  # [224,224,3]
    img224 = torch.from_numpy(raw).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
    my_pv = vla_diff.preprocess(img224).float().cpu()

    # (A) preprocessing match
    diff = (my_pv - real_pv).abs()
    cos = torch.nn.functional.cosine_similarity(
        my_pv.flatten(), real_pv.flatten(), dim=0
    ).item()
    print("=" * 70)
    print("(A) PREPROCESS MATCH  my_pv vs real pixel_values")
    print(f"    shapes: mine={tuple(my_pv.shape)} real={tuple(real_pv.shape)}")
    print(f"    max|diff|={diff.max():.4f}  mean|diff|={diff.mean():.5f}  cosine={cos:.6f}")

    # (B) token reproduction (teacher-forced argmax == greedy tokens)
    prompt_ids = inputs["input_ids"].to(DEVICE)
    tgt = ref_tokens.view(1, 7).to(DEVICE)
    with torch.no_grad():
        logits = vla_diff.action_token_logits(model, my_pv.to(DEVICE), prompt_ids, tgt)
    my_tokens = logits.argmax(-1)[0].cpu()
    match = int((my_tokens == ref_tokens).sum())
    print("(B) TOKEN REPRODUCTION  teacher-forced argmax vs greedy generate")
    print(f"    ref_tokens = {ref_tokens.tolist()}")
    print(f"    my_tokens  = {my_tokens.tolist()}")
    print(f"    exact match: {match}/7")
    print("=" * 70)
    ok_a = cos > 0.999 and diff.max() < 0.15
    ok_b = match == 7
    print(f"VERDICT: preprocessing {'OK' if ok_a else 'MISMATCH'}; "
          f"forward {'OK' if ok_b else 'MISMATCH'}")


if __name__ == "__main__":
    main()
