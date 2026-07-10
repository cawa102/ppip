# Literature research — targeted adversarial patch attacks vs OpenVLA/LIBERO

Background web research (2026-07-06) to inform the patch attack. Dense recipe + gotchas below.

## Bottom line
- **Test-time patch → full targeted *task* hijack of vanilla OpenVLA is essentially unsolved
  publicly.** Untargeted DoS is saturated (~100%: UADA/TMA/EDPA/FreezeVLA). Reliable *targeted*
  substitution is shown only via **training-time backdoors** (BackdoorVLA: OpenVLA LIBERO-Object
  **100%**, but 4% poison — out of scope) or **CoT hijack** (TRAP — needs a CoT stage OpenVLA lacks).
- So our prior "denial not hijack" matches the field. A positive test-time-patch hijack would be
  **novel**; libero_object is the most feasible suite (short horizon).

## Recipe (implement this)
1. **Teacher = behavioral distillation.** Roll out **task B (target) clean** to collect
   `(state_t, target 7 action-token ids)` pairs. (TRAP §C.2 does exactly this.)
2. **Loss = cross-entropy on the 7 action-token logits** toward the teacher token ids (TMA `L_TMA`,
   TRAP `L_action`). Optionally + small **expected-bin L2** (`Σ_j j·softmax(logit_j)`) for stability.
   MSE on de-tokenized continuous action FAILS (argmax non-differentiable).
3. **One static patch**, persists across the whole closed loop; optimize over a **batch of rollout
   states** (state coverage is the hard part — scene changes over 100–280 steps).
4. **EoT every step:** affine rotation ±20–30°, scale 0.7–1.3, translation, brightness/contrast;
   TV-smoothness loss; **the 0.9 center-crop is a MANDATORY EoT transform**.
5. **Hyperparams:** patch **10–30% area** (targeted needs bigger than the 5% untargeted norm),
   central & inside the 0.9 crop; **Adam lr ~1e-2, pixels unconstrained [0,1]**, ~1000–2000 iters,
   batch 8–16 states. (Alt: PGD L∞ step 2–8/255.)
6. **Adjudicate by closed-loop rollout** (target object in basket), NOT token-match — targeted
   numbers collapse at the long-horizon step.

## OpenVLA gotchas (verified against code — these bite)
- **Action token ids ∈ [31744, 31999]:** `id = vocab_size(32000) − bin`, `bin ∈ [1,256]`;
  decode `bin = 32000 − id`, `clip(bin−1,0,254)`. CE targets must be ids in this range.
- **Token 29871 = SentencePiece leading-space "▁", appended before actions** by `predict_action`
  (`if input_ids[:,-1]!=29871: append`). The optimization forward pass **must append 29871 exactly**
  or the 7 action logits are at the wrong positions → garbage gradients. (It is NOT an action token.)
- Greedy decode = **last 7 tokens**; teacher-force those 7 positions, CE vs target 7 ids.
- **center_crop=0.9** then resize-224 at eval; composite the patch consistently (keep it inside the
  central 90%) and treat the crop as EoT.
- **Fused encoder** (SigLIP+DINOv2) has **two different normalizations**; optimize in **pixel space**
  and let autograd flow through both branches.

## Key papers / code
- TMA/UADA/UPA — 2411.13587 (CE-on-tokens targeted recipe). EDPA — 2510.13237 (edpa-attack.github.io,
  untargeted feature disruption). FreezeVLA — 2509.19870 (freeze). TRAP — 2603.23117 (CoT hijack,
  behavioral distillation + homography EoT + TV). BackdoorVLA/AttackVLA — 2511.12149 (train-time
  targeted, 100% LIBERO-Object). Trajectory redirection — 2606.12978 (prompt, not patch).

## Implication for our design
Our behavioral-distillation + digital camera-space patch (Tier A) is the best-justified shot. Push:
big central patch, unconstrained pixels, CE-on-action-tokens teacher-forced with the 29871 append,
EoT incl. 0.9 crop, batch of rollout states. Adjudicate with the fixed closed-loop predicate.
