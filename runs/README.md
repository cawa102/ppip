# Runs

Local run artifacts go here.

Suggested structure:

```text
runs/
  pilot-001/
    candidates/
    metrics/
    ledger.jsonl
    summary.md
```

Raw rollout videos, large logs, and model outputs should not be committed if this workspace becomes a git repository.

## Long-Running Evaluation

OpenVLA+LIBERO candidate evaluation is expected to run longer than the 5-minute cap used by the original `karpathy/autoresearch` nanochat example.

Use the run ledger as the source of truth:

- submitted candidate;
- evaluation job ID or process metadata;
- budget stage;
- start/end timestamps;
- metrics path;
- failure reason if any;
- immutable official score.

A later Codex or Claude Code session should be able to resume by reading this directory and continuing from the last completed ledger entry.
