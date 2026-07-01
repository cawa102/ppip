# Risk Register

## R1: OpenVLA does not obey visual prompts

**Impact:** Targeted substitution may not appear in the initial task set.

**Mitigation:** Treat task denial and vision-language conflict characterization as secondary outcomes. Start with simpler task pairs and visible, semantically direct prompts.

## R2: Autonomous loop games the metric

**Impact:** Results become invalid.

**Mitigation:** Keep evaluator code read-only during benchmark runs. Save raw counts and candidate files. Compare against human and random baselines.

## R3: Rollouts are too expensive

**Impact:** Too few candidates can be evaluated.

**Mitigation:** Use a cheap pilot first. Reduce task count and rollout count before adding search sophistication.

## R4: Novelty overlap with SABER

**Impact:** Contribution may look incremental.

**Mitigation:** Emphasize environmental visual prompt discovery and scaffold capability evaluation, not text-instruction perturbation.

## R5: Candidate rendering becomes the bottleneck

**Impact:** Research time shifts from security question to graphics plumbing.

**Mitigation:** Start with simple text planes/labels in a limited set of placements before object-level texture editing.

## R6: Safety framing is misunderstood

**Impact:** The work may look like real-world robot attack enablement.

**Mitigation:** State simulation-only boundaries in every project-facing document and avoid physical deployment instructions.
