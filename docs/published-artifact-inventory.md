# Published artefact inventory

What is currently published, where, and what is deliberately not. Generated from the pushed state,
not from notes.

| | |
|---|---|
| **Repository** | https://github.com/cawa102/vla-visual-injection |
| **Visibility** | **private** — matches the dissertation's Data Availability statement (shared with supervisor and moderator, not a public release) |
| **Branch** | `submission/ele8095` (also the default branch) |
| **Commit** | `6837485263af4c8b43ed7602dd3a6fee785ad30d` — single orphan commit, **no history** |
| **Pushed** | 2026-08-25 |
| **Contents** | 732 files, 13 MB |
| **Media** | none in git — see the Google Drive set below |

The branch is an orphan: it shares no history with `cawa102/ppip`, so none of that repository's
internal documents or media blobs are reachable from it.

## At a glance

| Path | Files | What |
|---|---:|---|
| `experiments/patch_attack/` | 54 | the attack method — per-frame forcing, confinement, rendered proxy, logo ladder, word gate, static baselines |
| `experiments/configs/` | 4 | candidate schema, evaluation budgets, baseline and condition definitions |
| `experiments/` (other) | 5 | `run_candidate.py`, `results/aggregate_results.py`, example candidates |
| `src/evaluator/` | 10 | the fixed evaluation path: predicate, metrics, OpenVLA rollout backend |
| `src/rendering/` | 6 | scene injection and the in-scene display |
| `src/candidate_ledger/` | 4 | candidate emission and the append-only ledger |
| `runs/` | 645 | recorded results — 427 JSON, 135 log, 75 JSONL, 9 shell invocations |
| `third_party/pinned-versions.txt` | 1 | pinned OpenVLA and LIBERO commit hashes |
| `pyproject.toml`, `uv.lock`, `.gitignore` | 3 | package versions |

By type: 427 `.json`, 135 `.log`, 76 `.py`, 75 `.jsonl`, 9 `.sh`, 3 `.yaml`, 3 `.tsv`, 1 `.txt`,
1 `.toml`, 1 `.lock`.

**No `.md` files.** The work is described by the dissertation and the presentation.

## Directory names differ from this working copy

The published tree was renamed. When moving between the two, translate:

| here (working copy) | published |
|---|---|
| `runs/autoresearch-hijack/` | `runs/perframe-forcing/` |
| `runs/autoresearch-goal/` | `runs/readable-injection-goal/` |
| `runs/autoresearch-jul3/` | `runs/readable-injection-jul3/` |
| `src/autoresearch_loop/` | `src/candidate_ledger/` |
| `$HOME/autoresearch` (repo root) | `$HOME/vla-visual-injection` |
| `/home/40473058@eeecs.qub.ac.uk/…` | `/home/user/…` |

The repository **must be cloned as `vla-visual-injection`** — 54 modules insert
`$HOME/vla-visual-injection/src` on `sys.path`, so a differently named clone will not import.

Citations to `karpathy/autoresearch` were kept: that is the upstream open-source project
`run_candidate.py` and `results_tsv.py` were ported from, and removing it would leave an
uncredited port.

## Where each reported result's evidence sits

| Dissertation item | Published path |
|---|---|
| 12/12 jitter (Abstract, §V-B) | `runs/perframe-forcing/hitrate/` |
| 7/10 states, all of Table III | `runs/perframe-forcing/generalize/` |
| Table III clean target-policy column | `runs/perframe-forcing/base_policy/s0_inits0-9.log` |
| Table II readable + non-legible candidates (41 + 6 + 14 controls) | `runs/readable-injection-goal/`, `runs/readable-injection-jul3/`, `…/level3/` — ledgers + candidate and metrics JSONs |
| Table II static white-box row | `runs/perframe-forcing/candidates/*/rollouts.jsonl`, `runs/perframe-forcing/logs/eval_*.log` |
| Table IV corner sweep (latch, imperfect steps) | `runs/monitor-corner/result_corner_BL_*.json` + matching `trace_*.json` |
| Table V matched controls | `runs/monitor-corner/result_corner_BL_64_seed0_ctl_*.json` |
| §IV-C non-overlap over 21 layouts | `runs/monitor-stealth/occlusion.json` |
| §V-D 81% agreement, 37.9% / 26.3% areas | `runs/monitor-stealth/ladder/` |
| Table VI ε ladder, LPIPS, churn | `runs/monitor-stealth/ladder_hinge/` |
| §V-D ε = 0.09 held-out (1/12, 1/12, 10/12) | `runs/monitor-stealth/asr_eps009/` |
| §III-B clean target policy 11/12, four alternatives 0/12 | `runs/monitor-stealth/ceiling/` |
| Table VII paired comparison, 99.5% of 4,866 steps | `runs/monitor-stealth/word-gate/stage_c/`, `…/analysis/` |
| §V-E word-alone control | `runs/monitor-stealth/word-gate/word_alone/` |
| §V-E λ ablation (stratified + 1,471-frame check) | `runs/monitor-stealth/word-gate/lam*` |
| Table VIII replay and transfer | `runs/monitor-stealth/word-gate/artifact/init*/`, `…/transfer_from46/`, `…/replay_init46/` |
| §V-E insertion-slot results | `runs/monitor-stealth/word-gate/slot_1…slot_11/`, `…/artifact/crosspos_h400/` |
| §V-C rendered-display boundary | `runs/monitor-render/render_h11_s65_tex128.log`, `render_seed0_emis.log` |
| §IV-B / §VI-D invoked optimiser settings | `runs/perframe-forcing/run_primary_attack.sh` |

## Not in the repository

| Excluded | Why | Where instead |
|---|---|---|
| ~79,600 per-frame PNGs, `.pkl` checkpoints, `.npy`/`.pt` weights (~6 GB) | regenerable intermediates; the paper says these are retained rather than distributed | not distributed |
| all demo video | too large for git | Google Drive (below) |
| `runs/monitor-crosstask/`, `monitor-patch/`, `monitor-hijack/`, `pilot-00*`, `smoke-001`, `tomato-sauce-*` | no reported number depends on them; the paper mentions no cross-task pair | not distributed |
| exploratory sweeps (`kappa_sweep`, `objective_compare`, `objective_probe`, `perframe`, `asr_eps0.06`, `word-gate/stage_b`) | not reported | not distributed |
| `artifact/crosspos/` (~160-step sweep) | superseded 2026-08-18 by the horizon-400 re-measurement, which is published | not distributed |
| unit tests, planning documents, working log, design notes under `docs/research/` | not results; the design notes also contradict the final paper (they call white-box gradients out of scope) | not distributed |

## Companion media set (Google Drive)

Staged at `…/scratchpad/drive-upload/` — **561 MB, 27 GIFs + 3,045 PNGs**, all GIF (no MP4, so it
plays anywhere). Organised by dissertation section:

| Folder | Size | Contents |
|---|---:|---|
| `01-primary-attack/` | 103 MB | 3 demo GIFs + 309 frames for Figure 1 (clean vs attacked) |
| `02-confinement/` | 281 MB | 7 size-sweep GIFs, 3 control GIFs, 1,099 policy-input frames for Figure 4 |
| `03-render-boundary/` | 24 MB | `exp1_confined_patch_success.gif`, `exp2_render_boundary.gif` (named in the Figure 5 note) |
| `04-logo-ladder/` | 97 MB | 8 ladder GIFs + patch crops for Figure 6 |
| `05-heldout-transfer-eps009/` | 11 MB | `heldout_transfer.gif` |
| `06-word-gate/` | 23 MB | three-way, initial, and replay GIFs |
| `07-readable-baselines/` | 26 MB | 455 candidate keyframes for Figure 2 |

## Verified before publishing

- 427/427 JSON parse; 122/122 ledger `candidate_path`/`metrics_path` targets resolve after the rename
- no unresolved local imports across all 76 Python files
- 28/28 reported results have their evidence present
- 0 occurrences of the student number, of any AI-tool name, and of media or binary blobs

## Open items

1. The dissertation's Figures 1, 2, 4, 5 and 6 are still placeholders; the Drive set holds the frames
   needed to build them. Figure 3 points at `figures/attack-pipeline-revised.png`, which does not exist.
2. No code reproduces the Wilson intervals or the exact McNemar test reported in §V-E.
3. **`cawa102/ppip` is still public** and its `main` still carries the internal guidance files,
   `docs/research/research-log.md`, `docs/plans/`, `programs/`, `tests/` and the student number.
   Deleting files there would not remove them from history — the effective options are making the
   repository private or deleting it.
