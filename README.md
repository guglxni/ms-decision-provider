# ms-decision-provider

**Microsoft-Decision-1 (Azure AI Foundry "System One") decision provider for Hermes Agent** — typed, probabilistic decisions for classification, routing, and prioritization, with an optional `pre_tool_call` safety gate and a **shadow replay harness**.

This is a community reference implementation of the **typed probabilistic decision-provider runtime** proposed in **[NousResearch/hermes-agent#113008](https://github.com/NousResearch/hermes-agent/issues/113008)**: bounded semantic questions whose output space is known in advance, where the useful result is a **probability distribution** rather than prose.

## Why

Hermes (and agents generally) make many small semantic decisions inside otherwise deterministic workflows — *is this tool call safe to run unattended? which model should handle this task? how severe is this incident?* Today those go through full generative LLM calls: slow, expensive, and they collapse a probability distribution into prose.

A decision-native backend is different: the output domain is declared up front, **probabilities survive the provider boundary**, and the caller can **abstain** below a threshold. Microsoft's `Microsoft-Decision-1` (GA in Azure AI Foundry) is exactly that — a System One classifier you call at `POST {host}/providers/microsoft/v1/systemone`, answering three typed question shapes:

| Type | Question shape | Returns | Use for |
|------|----------------|---------|---------|
| `noul` | yes/no | `{"noul": 0.93}` probability | gating, safety, binary classification |
| `choice` | pick from a fixed set | choice + per-option probabilities + confidence | routing (model/team/queue) |
| `score` | ordered scale | continuous score + distribution + legend | prioritization, severity |

**Decision output is evidence, not authority.** The plugin preserves the full distribution; the *consumer* owns the threshold and the failure policy.

## What it adds

### Tools (registered when `AZURE_FOUNDRY_API_KEY` is present)

- **`decision_noul`** — `state`, `question`, optional `true_label`/`false_label` → yes/no probability.
- **`decision_choice`** — `state`, `question`, `options[]`, optional `criteria{option:desc}` → chosen option + probabilities + confidence.
- **`decision_score`** — `state`, `question`, `levels[]` (low→high) → continuous score + distribution + legend.
- **`decide`** — `state`, `questions{id:{type,instructions,criteria}}` → batched typed decisions in one call.

### Optional safety gate (`pre_tool_call` hook)

When `MS_DECISION_GATE=1`, a `pre_tool_call` hook classifies risky tool calls (`terminal`, `write_file`, `patch`, `process_manage`, browser actions, `cronjob_manage`, `delegate_task`, `send_message`, `skill_manage`) and **blocks** any the model judges unsafe to run unattended (safety probability below `MS_DECISION_BLOCK_THRESHOLD`, default `0.5`). Reads and the decision tools themselves pass through unclassified. The gate **fails open** on any provider error so a down/rate-limited endpoint never wedges the agent loop.

```json
{"action": "block",
 "message": "[ms-decision-gate] Microsoft-Decision-1 judged this terminal call unsafe to auto-run (safety p=0.00 < 0.5). Needs explicit user approval."}
```

### Shadow replay harness (`eval/`)

`eval/replay.py` replays a labeled decision corpus through the provider **without changing Hermes behavior** and scores each family independently — the RFC's shadow-mode / evaluation acceptance criterion. See [`eval/README.md`](eval/README.md).

**Measured on the shipped 32-case corpus** (GlobalStandard, capacity 100):

| Family | Type | Metric | Result |
|--------|------|--------|--------|
| `tool_safety` | noul | accuracy / Brier / coverage@0.7 | **1.00** / **0.012** / 0.92 → 1.00 acc when confident |
| `model_routing` | choice | top-1 accuracy / mean conf | **1.00** / 0.99 |
| `incident_severity` | score | exact / adjacent / MAE | **0.90** / **1.00** / 0.139 |

Latency p50 ≈ 1.0–1.2 s; ~40–110 input tokens, **1–3 output tokens** per call.

## Prerequisites

1. An Azure AI Foundry resource with a **`Microsoft-Decision-1`** deployment (`Microsoft` format, capability `decision: true`):
   ```bash
   az cognitiveservices account deployment create \
     -g <resource-group> -n <foundry-resource> \
     --deployment-name ms-decision-1 \
     --model-name Microsoft-Decision-1 --model-version 1 \
     --model-format Microsoft --sku-name GlobalStandard --sku-capacity 100
   ```
   Capacity 1 rate-limits almost immediately; use a comfortable capacity.
2. `AZURE_FOUNDRY_API_KEY` (and `AZURE_FOUNDRY_BASE_URL`) in `~/.hermes/.env`.

> **OpenAI's `/v1/decisions` is a different, OpenAI-platform-only beta** and is **not** exposed on Azure AI Foundry (verified 404 across path shapes). `Microsoft-Decision-1` is the Foundry-native equivalent and what this plugin targets.

## Install

Until catalog-listed, install as a directory plugin:

```bash
git clone https://github.com/guglxni/ms-decision-provider ~/.hermes/plugins/ms-decision-provider
hermes plugins doctor ~/.hermes/plugins/ms-decision-provider
```

Once catalog-listed: `hermes plugins install ms-decision-provider`.

## Configuration

| Env var | Default | Meaning |
|---------|---------|---------|
| `AZURE_FOUNDRY_API_KEY` | — (required) | Foundry API key |
| `AZURE_FOUNDRY_BASE_URL` | `<your-foundry-resource>` | Resource base URL (`/openai/v1` suffix stripped — System One hangs off the host root) |
| `MS_DECISION_MODEL` | `ms-decision-1` | Deployment name |
| `MS_DECISION_GATE` | `0` (off) | Set `1` to enable the `pre_tool_call` gate |
| `MS_DECISION_BLOCK_THRESHOLD` | `0.5` | Safety probability below which the gate blocks |
| `MS_DECISION_TIMEOUT` | `20` | Gate call timeout (s) |

## Design notes (aligned with RFC #113008)

- **Probabilities are first-class** — full distributions returned, never collapsed to a label by the provider.
- **Consumer owns policy** — thresholds and abstention live at the call site, not in a global constant.
- **Explicit state projection** — the provider receives only the `state` string the caller supplies; it does not inherit session history, memory, system prompts, or credentials.
- **Abstention is normal** — low-confidence or provider-unavailable is distinguishable from "decision made" and defaults to existing behavior (fail open).
- **Shadow evaluation is first-class** — `eval/replay.py` scores each decision family (accuracy, Brier, calibration, risk-vs-coverage) without touching runtime behavior.
- **No prompt-cache impact** — registration adds no model-visible system-prompt text; tools are ordinary registry entries.
- **No TypeSafe/Jev dependency** — provider-neutral in spirit; this is one backend (Microsoft-Decision-1 on Azure), comparable against others via the replay harness.

## Failure semantics

Provider disabled / missing credential / timeout / malformed response → **existing Hermes behavior** (the gate allows, tools return `{"error": ...}`). Fail-closed is the caller's explicit choice at its own enforcement boundary, not a silent default.

## License

MIT
