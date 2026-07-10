"""Stage 1 — collect teacher states for behavioral distillation.

Roll out the TARGET task (salad_dressing) clean and collect every agentview frame (the states
along the desired trajectory). For each frame compute the teacher's greedy 7 action tokens =
OpenVLA(frame, TARGET). Save (images, teacher_tokens) so the optimizer can push
OpenVLA(frame+patch, USER) toward these.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HOME = os.path.expanduser("~")
for p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, p))

import patch_config as C  # noqa: E402
import vla_diff  # noqa: E402
from hijack_backend import HijackBackend  # noqa: E402

SEEDS = [
    int(seed)
    for seed in os.environ.get("PATCH_ATTACK_SEEDS", "0,1").split(",")
    if seed.strip()
]
SUBSAMPLE = int(os.environ.get("PATCH_ATTACK_SUBSAMPLE", "2"))
MAX_FRAMES = int(os.environ.get("PATCH_ATTACK_MAX_FRAMES", "0"))
MAX_STEPS = int(os.environ.get("PATCH_ATTACK_MAX_STEPS", "180"))
TEACHER_BATCH = int(os.environ.get("PATCH_ATTACK_TEACHER_BATCH", "4"))


def _carrier(cid: str) -> dict:
    return {
        "candidate_id": cid,
        "condition": "loop_with_skill",
        "user_task": C.USER_TASK,
        "target_task": C.TARGET_TASK,
        "visual_prompt": {"text": "(offcamera carrier)", "prompt_level": "optimized_typographic_prompt"},
        "placement": {"scene_anchor": "table_surface", "position": [0.9, 0.5, 1.9],
                      "rotation": [90.0, 90.0, 0.0], "scale": 0.6},
        "style": {"foreground_color": "#000000", "background_color": "#ffffff",
                  "font_size": 40, "font_family": "sans-serif"},
        "metadata": {"created_by": "hijack_collect", "created_at": "2026-07-06T00:00:00Z",
                     "notes": "teacher-state collection (target instruction)."},
    }


def teacher_tokens(
    model, processor, frames: np.ndarray, instruction: str, batch_size: int = TEACHER_BATCH
) -> np.ndarray:
    """Greedy 7 action-token ids of OpenVLA(frame, instruction) for each frame.

    The pinned OpenVLA generation helper raises on batch size > 1. We avoid it
    by greedily decoding through the verified multimodal-embedding path in
    `vla_diff.py`, with no KV cache and no gradients.
    """
    dev = "cuda"
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    ids = processor.tokenizer(prompt, return_tensors="pt").input_ids
    if ids[0, -1].item() != vla_diff._SPACE_TOKEN:
        ids = torch.cat([ids, torch.tensor([[vla_diff._SPACE_TOKEN]], dtype=ids.dtype)], dim=1)
    out: list[np.ndarray] = []
    for i in range(0, len(frames), batch_size):
        chunk = frames[i : i + batch_size]
        img = torch.from_numpy(chunk.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)
        pv = vla_diff.preprocess(img).to(model.dtype)
        bids = ids.repeat(len(chunk), 1).to(dev)
        with torch.no_grad():
            full = vla_diff.build_multimodal_embeds(model, pv, bids)
            generated: list[torch.Tensor] = []
            for _ in range(vla_diff.ACTION_DIM):
                logits = model.language_model(inputs_embeds=full, use_cache=False).logits
                next_token = logits[:, -1, :].argmax(dim=-1)
                generated.append(next_token.detach().cpu())
                next_embed = model.get_input_embeddings()(next_token[:, None])
                full = torch.cat([full, next_embed], dim=1)
        out.append(torch.stack(generated, dim=1).numpy())
        del img, pv, bids, full
        torch.cuda.empty_cache()
        done = min(i + len(chunk), len(frames))
        print(f"[collect] teacher tokens {done}/{len(frames)}", flush=True)
    return np.concatenate(out, axis=0)


def action_tokens_from_raw_actions(model, actions: np.ndarray, unnorm_key: str) -> np.ndarray:
    """Invert OpenVLA's action-token decode for raw continuous policy actions."""
    stats = model.get_action_stats(unnorm_key)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.where(mask, 2.0 * (actions - low) / (high - low) - 1.0, actions)
    normalized = np.clip(normalized, -1.0, 1.0)
    discretized = np.digitize(normalized, np.asarray(model.bins, dtype=np.float32))
    tokens = int(model.vocab_size) - discretized
    return tokens.astype(np.int64)


def main() -> None:
    backend = HijackBackend(run_dir=C.RUN_DIR, max_steps=MAX_STEPS)  # cap rollout length for speed
    backend.set_instruction_override(C.TARGET_TASK)  # command salad_dressing
    all_frames: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    for s in SEEDS:
        backend._collect = []
        backend._collect_actions = []
        backend.run_rollouts(candidate=_carrier(f"collect_s{s}"), seeds=[s], rollouts_per_candidate=1)
        # Cap to the grasp-and-place phase; the post-success tail is off-task noise.
        frames = backend._collect[:MAX_STEPS][::SUBSAMPLE]
        actions = backend._collect_actions[:MAX_STEPS][::SUBSAMPLE]
        all_frames.extend(frames)
        all_actions.extend(actions)
        print(f"[collect] seed {s}: {len(backend._collect)} frames -> kept {len(frames)}", flush=True)

    frames = np.stack(all_frames).astype(np.uint8)  # [N,224,224,3]
    actions_arr = np.stack(all_actions).astype(np.float32)  # [N,7], raw predict_action outputs
    if MAX_FRAMES > 0:
        frames = frames[:MAX_FRAMES]
        actions_arr = actions_arr[:MAX_FRAMES]
        print(f"[collect] truncated to first {len(frames)} frames by PATCH_ATTACK_MAX_FRAMES", flush=True)
    frames_only = os.path.join(C.RUN_DIR, "states_saladdressing_frames_only.npz")
    np.savez_compressed(frames_only, images=frames, actions=actions_arr)
    print(f"[collect] saved frames-only checkpoint -> {frames_only}", flush=True)
    model, processor, _, _ = backend._policy
    tokens = action_tokens_from_raw_actions(model, actions_arr, backend.unnorm_key)
    np.savez_compressed(C.STATES_NPZ, images=frames, actions=actions_arr, tokens=tokens)
    print(f"[collect] saved {len(frames)} states -> {C.STATES_NPZ}", flush=True)
    print(f"[collect] token id range: [{tokens.min()}, {tokens.max()}] (expect 31744..31999)", flush=True)


if __name__ == "__main__":
    main()
