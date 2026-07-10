"""Instrument the real OpenVLA-7B forward pass and dump the intermediate tensors:

  raw image (PNG on disk)
    -> processor        -> pixel_values          [1, 6, 224, 224]  (SigLIP+DINOv2 stacked)
    -> vision_backbone  -> patch_features         [1, 256, 2176]    (ENCODER output)
    -> projector (MLP)  -> projected_patch_embeds [1, 256, 4096]    (PROJECTOR output)
    -> cat(BOS, patches, text) = multimodal_embeds [1, 1+256+T, 4096] (LLAMA-2 input)
    -> Llama-2 7B       -> 7 action tokens        -> 7-DoF action

Uses the genuine production path (backend._load_policy + reference get_vla_action);
the four capture points are populated by forward hooks, nothing is re-implemented.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch", "src"))
sys.path.insert(0, os.path.join(HOME, "openvla"))

from evaluator.openvla_backend import OpenVLARolloutBackend  # noqa: E402
from experiments.robot.openvla_utils import get_vla_action  # noqa: E402

TASK_LABEL = "pick up the alphabet soup and place it in the basket"
CLEAN_IMG = os.path.join(
    HOME,
    "autoresearch/runs/autoresearch-goal/candidates/"
    "g_r2b_base_alphabet_soup/seed0_ep0_first.png",
)
INJECTED_IMG = os.path.join(
    HOME,
    "autoresearch/runs/autoresearch-goal/candidates/"
    "g_r5_dos_alphabet_soup_confirm/seed0_ep0_first.png",
)
OUT_DIR = os.path.join(HOME, "autoresearch/runs/analysis/encoder-probe-001")
os.makedirs(OUT_DIR, exist_ok=True)


def _stats(t: torch.Tensor) -> str:
    f = t.float()
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} "
        f"mean={f.mean():+.4f} std={f.std():.4f} "
        f"min={f.min():+.3f} max={f.max():+.3f}"
    )


def run_once(model, processor, backend, img_path: str, captured: dict) -> np.ndarray:
    captured.clear()
    obs = {"full_image": np.array(Image.open(img_path).convert("RGB"))}
    with torch.no_grad():
        action = get_vla_action(
            model, processor, backend.model_id, obs, TASK_LABEL,
            backend.unnorm_key, center_crop=True,
        )
    return action


def main() -> None:
    backend = OpenVLARolloutBackend()  # finetuned-libero-object, center_crop=True
    print(f"[load] {backend.model_id}  (bf16, sdpa) ...", flush=True)
    model, processor, cfg, resize_size = backend._load_policy()
    model.eval()
    print(f"[load] done. resize_size={resize_size}", flush=True)

    captured: dict = {}

    def cap_model_pre(module, args, kwargs):
        pv = kwargs.get("pixel_values")
        if pv is None and len(args) >= 3:
            pv = args[2]
        if pv is not None and "pixel_values" not in captured:
            ids = kwargs.get("input_ids")
            if ids is None and len(args) >= 1:
                ids = args[0]
            captured["pixel_values"] = pv.detach().float().cpu()
            if ids is not None:
                captured["input_ids"] = ids.detach().cpu()

    def cap_llm_pre(module, args, kwargs):
        ie = kwargs.get("inputs_embeds")
        if ie is not None and ie.shape[1] > 1 and "llama_input" not in captured:
            captured["llama_input"] = ie.detach().float().cpu()

    def cap_encoder(m, i, o):  # MUST return None, else the output is replaced
        captured.setdefault("encoder_out", o.detach().float().cpu())

    def cap_projector(m, i, o):
        captured.setdefault("projector_out", o.detach().float().cpu())

    model.register_forward_pre_hook(cap_model_pre, with_kwargs=True)
    model.vision_backbone.register_forward_hook(cap_encoder)
    model.projector.register_forward_hook(cap_projector)
    model.language_model.register_forward_pre_hook(cap_llm_pre, with_kwargs=True)

    results = {}
    for tag, path in (("clean", CLEAN_IMG), ("injected", INJECTED_IMG)):
        print(f"\n[run:{tag}] {path}", flush=True)
        action = run_once(model, processor, backend, path, captured)
        snap = {k: v.clone() for k, v in captured.items()}
        snap["action"] = torch.tensor(np.asarray(action, dtype=np.float32))
        results[tag] = snap
        # persist (float16 to keep it small; input_ids as int)
        for k, v in snap.items():
            np.save(
                os.path.join(OUT_DIR, f"{tag}_{k}.npy"),
                v.numpy().astype(np.float16 if k != "input_ids" else np.int64),
            )

    # ---- report ----
    tok = processor.tokenizer
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    c = results["clean"]
    emit("=" * 78)
    emit("PRIMARY EXAMPLE  (one image -> full forward pass)")
    emit("=" * 78)
    emit(f"raw image file : {CLEAN_IMG}")
    emit(f"                 (224x224 RGB agentview frame, clean alphabet_soup scene)")
    emit(f"instruction    : {TASK_LABEL!r}")
    emit(f"prompt fed     : 'In: What action should the robot take to "
         f"{TASK_LABEL.lower()}?\\nOut:'")
    emit("")
    ids = c["input_ids"][0].tolist()
    emit(f"[input_ids]        {tuple(c['input_ids'].shape)}  ({len(ids)} text tokens "
         f"incl. BOS + appended empty-token 29871)")
    emit(f"                   decoded: {tok.decode(ids)!r}")
    emit(f"[pixel_values]     {_stats(c['pixel_values'])}")
    emit("                   ^ 6 channels = SigLIP(3) + DINOv2(3), each own normalization")
    emit("")
    emit(f"[1] ENCODER  out   {_stats(c['encoder_out'])}")
    emit(f"                   = cat(SigLIP 256x1152, DINOv2 256x1024) -> 256 patches x 2176")
    emit(f"                   patch[0,0,:6] = "
         f"{c['encoder_out'][0,0,:6].numpy().round(4).tolist()}")
    emit("")
    emit(f"[2] PROJECTOR out  {_stats(c['projector_out'])}")
    emit(f"                   = 3-layer MLP (2176->8704->4096->4096) -> 256 patches x 4096")
    emit(f"                   patch[0,0,:6] = "
         f"{c['projector_out'][0,0,:6].numpy().round(4).tolist()}")
    emit("")
    T = c["llama_input"].shape[1]
    emit(f"[3] LLAMA-2 input  {_stats(c['llama_input'])}")
    emit(f"                   seq_len {T} = 1 (BOS) + 256 (patch tokens) + "
         f"{T-257} (text tokens)")
    emit(f"                   layout: [BOS | 256 projected patches | text instruction]")
    # verify the patch block of llama_input equals projector_out
    same = torch.allclose(
        c["llama_input"][:, 1:257, :], c["projector_out"], atol=1e-3
    )
    emit(f"                   llama_input[:,1:257,:] == projector_out ? {same}")
    emit("")
    emit(f"[out] 7-DoF action : {c['action'].numpy().round(4).tolist()}")
    emit(f"                     (dx,dy,dz, droll,dpitch,dyaw, gripper)")

    # ---- bonus: clean vs injected at each stage (the DoS mechanism, for free) ----
    inj = results["injected"]

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        a2, b2 = a.flatten().float(), b.flatten().float()
        return float(torch.dot(a2, b2) / (a2.norm() * b2.norm() + 1e-8))

    emit("")
    emit("=" * 78)
    emit("BONUS  clean  vs  injected (in-view DoS label)  — same hooks, same instruction")
    emit("=" * 78)
    emit(f"injected image : {INJECTED_IMG}")
    for stage in ("pixel_values", "encoder_out", "projector_out", "llama_input"):
        emit(f"  cos(clean, injected) @ {stage:<14} = {cos(c[stage], inj[stage]):.4f}")
    # llama_input: patch block vs text block separately
    emit(f"  cos @ llama_input[patches 1:257]      = "
         f"{cos(c['llama_input'][:,1:257,:], inj['llama_input'][:,1:257,:]):.4f}")
    emit(f"  cos @ llama_input[text   257:]        = "
         f"{cos(c['llama_input'][:,257:,:], inj['llama_input'][:,257:,:]):.4f}")
    emit(f"  clean    action = {c['action'].numpy().round(4).tolist()}")
    emit(f"  injected action = {inj['action'].numpy().round(4).tolist()}")

    with open(os.path.join(OUT_DIR, "SUMMARY.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    emit("")
    emit(f"[saved] tensors + SUMMARY.txt under {OUT_DIR}")


if __name__ == "__main__":
    main()
