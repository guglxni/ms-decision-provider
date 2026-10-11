# eval/ — shadow replay harness

A labeled-decision replay harness implementing the **shadow-mode / evaluation** acceptance criteria from [NousResearch/hermes-agent#113008](https://github.com/NousResearch/hermes-agent/issues/113008): *evaluate decision families, not models globally; abstention is first-class; thresholds are owned by the family; measure whether acting on the decision improves the outcome.*

It replays a labeled corpus through Microsoft-Decision-1 **without changing any Hermes behavior** and scores each decision family independently.

## Run

```bash
python3 eval/replay.py                    # all families
python3 eval/replay.py --family tool_safety
python3 eval/replay.py --coverage 0.8     # abstain below p=0.8
```

Requires `AZURE_FOUNDRY_API_KEY` (env or `~/.hermes/.env`). **Fails closed** on provider error — a replay harness must never fabricate metrics.

## Metrics per family

- **`noul`** (binary) — top-1 accuracy, **Brier score**, **calibration by 0.1 bucket**, and **risk-vs-coverage** (accuracy over the subset where the model is confident enough to act at the chosen threshold).
- **`choice`** (fixed set) — top-1 accuracy, mean confidence, risk-vs-coverage, per-case picks.
- **`score`** (ordered scale) — exact-level accuracy, adjacent (off-by-one) accuracy, MAE, per-case scores.

## Corpus

`decision_cases.json` ships three families mirroring the plugin's real uses:

| Family | Type | Cases | Ground truth |
|--------|------|-------|--------------|
| `tool_safety` | noul | 12 | safe-to-auto-run yes/no |
| `model_routing` | choice | 10 | correct model for the task |
| `incident_severity` | score | 10 | severity level 0–3 |

Add cases by appending to a family's `cases` array. Labels are the reviewer's judgment of the *correct* decision, not the model's output.

## Comparing providers

To A/B Microsoft-Decision-1 against another backend (Jev, Laya, or an LLM via `typesafe-ai/system-one-adapter-python`), point the same corpus at the other provider behind the same `_call()` contract and diff the per-family metrics — this is the comparison the RFC asks for before baking any vendor into core.
