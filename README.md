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

### Tools (registered when both `AZURE_FOUNDRY_API_KEY` and `AZURE_FOUNDRY_BASE_URL` resolve in the active Hermes secret scope)

- **`decision_noul`** — `state`, `question`, optional `true_label`/`false_label` → yes/no probability.
- **`decision_choice`** — `state`, `question`, `options[]`, optional `criteria{option:desc}` → chosen option + probabilities + confidence.
- **`decision_score`** — `state`, `question`, `levels[]` (low→high) → continuous score + distribution + legend.
- **`decide`** — `state`, `questions{id:{type,instructions,criteria}}` → batched typed decisions in one call.

### Optional safety gate (`pre_tool_call` hook)

When `MS_DECISION_GATE=1`, a `pre_tool_call` hook sends **only a fixed tool name and coarse, allowlisted operation label** to Azure; raw arguments, file paths, messages, URLs, and other private contents stay local. For known-risk tools (`terminal`, `execute_code`, `write_file`, `patch`, `process_manage`, browser actions, `cronjob_manage`, `delegate_task`, `send_message`, `skill_manage`) **and unrecognized `mcp__*` calls**, the remote score is **advisory**. The hook requests Hermes's human approval for every such call (including missing/malformed provider responses); it does not auto-authorize based on a model score. The local human prompt shows the complete action arguments with Hermes's secret redactor forced on, while Azure receives only the coarse projection. In `MS_DECISION_GATE_MODE=block`, it hard-blocks when the score is below `MS_DECISION_BLOCK_THRESHOLD` (default `0.5`) or unavailable. Decision tools are excluded to avoid recursion; other tools follow Hermes's existing guardrails. This is **not** a universal security boundary or a replacement for Hermes's permissions.

`approve` uses Hermes's approval path, including its configured session/permanent allowlist, cron and single-query policies, and explicit yolo bypasses. An approval rule key is scoped to the full call via a per-process HMAC, rather than the tool name alone; a process restart invalidates previously cached rules. Users who require hard denial should select `MS_DECISION_GATE_MODE=block` and review Hermes's own approval settings. For this plugin, only the four typed decision tools explicitly send caller-supplied `state`/`questions` text to Azure. Do not put secrets or PII in those inputs unless the destination is authorized for them.

### Shadow replay harness (`tests/eval/`)

`tests/eval/replay.py` replays a labeled decision corpus through the provider **without changing Hermes behavior** and scores each family independently — the RFC's shadow-mode / evaluation acceptance criterion. See [`tests/eval/README.md`](tests/eval/README.md).

**Historical baseline** (32-case corpus before the `ts-07` wording was sanitized; GlobalStandard, capacity 100). These numbers are **not** a fresh measurement of the revised corpus or the privacy-projected gate. In particular, the gate intentionally maps both destructive and benign `rm` calls to the same remote state and requires human approval either way; these accuracy figures cannot be interpreted as gate effectiveness. Re-run a projected-state evaluation on your own deployment before selecting thresholds:

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
2. `AZURE_FOUNDRY_API_KEY` and `AZURE_FOUNDRY_BASE_URL` set (environment or your Hermes secret scope — resolved via `agent.secret_scope.get_secret`). Nothing is sent anywhere until both are set; unset endpoint → the plugin returns `{"error": "set AZURE_FOUNDRY_BASE_URL"}` and transmits nothing.

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
| `AZURE_FOUNDRY_BASE_URL` | — (required) | Your Foundry resource base URL (`/openai/v1` suffix stripped — System One hangs off the host root). No default; unset = nothing is sent. |
| `MS_DECISION_MODEL` | `ms-decision-1` | Deployment name |
| `MS_DECISION_GATE` | `0` (off) | Set `1` to enable the `pre_tool_call` gate |
| `MS_DECISION_BLOCK_THRESHOLD` | `0.5` | Safety probability below which **block mode** hard-denies; all known-risk calls otherwise require approval |
| `MS_DECISION_GATE_MODE` | `approve` | `approve` asks a human; `block` hard-denies low-score/missing signals, asks for approval otherwise |
| `MS_DECISION_TIMEOUT` | `20` | Gate call timeout (s) |

## Design notes (aligned with RFC #113008)

- **Probabilities are first-class** — full distributions returned, never collapsed to a label by the provider.
- **Consumer owns policy** — thresholds and abstention live at the call site, not in a global constant.
- **Explicit state projection** — typed decision tools transmit only `state`/`questions` supplied by the caller, and the gate transmits only coarse allowlisted operation labels; no session history, memory, prompts, raw tool arguments, or credentials in request bodies.
- **Abstention is normal** — low confidence or provider unavailability produces an error for tools and never auto-authorizes a known-risk tool call; the host's approval policy decides the result.
- **Shadow evaluation is first-class** — `tests/eval/replay.py` scores each decision family (accuracy, Brier, calibration, risk-vs-coverage) without touching runtime behavior.
- **No prompt-cache impact** — registration adds no model-visible system-prompt text; tools are ordinary registry entries.
- **No TypeSafe/Jev dependency** — provider-neutral in spirit; this is one backend (Microsoft-Decision-1 on Azure), comparable against others via the replay harness.

## Failure semantics

Missing/invalid endpoint or credential → no request; the tool returns a sanitized `{"error": ...}`. HTTP failures and redirects are sanitized too. The optional gate does not auto-allow known-risk calls on transport errors, malformed responses, or low confidence: it asks Hermes's approval gate (or hard-blocks in `block` mode). Hermes's explicit yolo, cron/single-query approval policy, and user allowlists still apply. This plugin is advisory, not a comprehensive security sandbox.

## License

MIT
