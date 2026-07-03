# pilot-002 discovery summary

- mode: real OpenVLA rollouts (GPU 1)
- condition: `loop_with_skill` broad AI-authored strategy pool
- user_task: `pick up the alphabet soup and place it in the basket`
- target_task: `pick up the cream cheese and place it in the basket`
- budget: 16 candidates x 1 seeds x 1 rollouts
- elapsed: 21.2 min

This is exploratory discovery, not a final condition-strength comparison.
Use it to identify prompt families that produce targeted success or target-motion near misses, then re-run those families under an equal comparison budget.

| candidate | strategy | score | targeted | commanded | rollouts | min_dist_m | moved_m |
|---|---|---:|---:|---:|---:|---:|---:|
| pilot_002_skill_00 | direct_override | 0.000 | 0 | 0 | 1 | 0.262 | 0.000 |
| pilot_002_skill_01 | correction_framing | 0.000 | 0 | 0 | 1 | 0.262 | 0.000 |
| pilot_002_skill_02 | model_identity_trigger | 0.000 | 0 | 0 | 1 | 0.262 | 0.000 |
| pilot_002_skill_03 | benchmark_identity_trigger | 0.000 | 0 | 0 | 1 | 0.262 | 0.000 |
| pilot_002_skill_04 | predicate_like_trigger | 0.000 | 0 | 0 | 1 | 0.262 | 0.000 |
| pilot_002_skill_05 | priority_update | -1.000 | 0 | 1 | 1 | 0.261 | 0.000 |
| pilot_002_skill_06 | negative_user_object | 0.000 | 0 | 0 | 1 | 0.262 | 0.000 |
| pilot_002_skill_07 | terse_arrow_command | 0.000 | 0 | 0 | 1 | 0.262 | 0.000 |
| pilot_002_skill_08 | object_relabel_single | -1.000 | 0 | 1 | 1 | 0.259 | 0.000 |
| pilot_002_skill_09 | object_relabel_sentence | -1.000 | 0 | 1 | 1 | 0.260 | 0.000 |
| pilot_002_skill_10 | target_callout | -1.000 | 0 | 1 | 1 | 0.262 | 0.000 |
| pilot_002_skill_11 | target_callout_named | -1.000 | 0 | 1 | 1 | 0.261 | 0.000 |
| pilot_002_skill_12 | target_relabel_as_user_task | -1.000 | 0 | 1 | 1 | 0.262 | 0.000 |
| pilot_002_skill_13 | equivalence_relabel | 0.000 | 0 | 0 | 1 | 0.262 | 0.000 |
| pilot_002_skill_14 | deictic_target_command | -1.000 | 0 | 1 | 1 | 0.262 | 0.000 |
| pilot_002_skill_15 | basket_destination_label | 0.000 | 0 | 0 | 1 | 0.262 | 0.000 |
