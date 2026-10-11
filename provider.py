"""Microsoft-Decision-1 client + Hermes tool/hook registration.

Calls Azure AI Foundry's System One endpoint:
    POST {base}/providers/microsoft/v1/systemone

Typed questions:
    noul   -> yes/no probability           {"noul": 0.93}
    choice -> pick from a fixed set        {"choice": "infra", "probabilities": {...}}
    score  -> ordered-scale score          {"score": 2.94, "legend": {...}, "probabilities": {...}}

Every handler returns a JSON string and preserves the full distribution so the
caller can apply its own threshold / abstention policy.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional

_DEFAULT_BASE = "https://guglaniaaryan-4736-resource.openai.azure.com"
_MODEL_ENV = "MS_DECISION_MODEL"
_DEFAULT_MODEL = "ms-decision-1"

# Tools the pre_tool_call gate considers risky enough to classify. Reads and
# the decision tools themselves always pass through unclassified.
RISKY_TOOLS = frozenset({
    "terminal", "execute_code", "write_file", "patch", "process_manage",
    "browser_navigate", "browser_click", "browser_type", "cronjob_manage",
    "delegate_task", "send_message", "skill_manage",
})
SAFE_TOOLS = frozenset({
    "read_file", "search_files", "web_search", "web_extract", "session_search",
    "skill_view", "skills_list", "todo_list", "memory", "vision_analyze",
    "text_to_speech", "decide", "decision_noul", "decision_choice", "decision_score",
})

# Gate policy knobs (env-overridable). The consumer owns the threshold.
_BLOCK_THRESHOLD = float(os.environ.get("MS_DECISION_BLOCK_THRESHOLD", "0.5"))
_TIMEOUT = float(os.environ.get("MS_DECISION_TIMEOUT", "20"))
_GATE_ENABLED = os.environ.get("MS_DECISION_GATE", "0") in ("1", "true", "yes")


def _env_file_key(key: str) -> str:
    """Read a credential from the OS env, falling back to ~/.hermes/.env."""
    val = os.environ.get(key)
    if val:
        return val
    env_path = os.path.expanduser("~/.hermes/.env")
    try:
        with open(env_path) as fh:
            for line in fh:
                if line.startswith(key + "="):
                    return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def _base_url() -> str:
    """System One hangs off the resource host root, NOT the .../openai/v1 suffix
    that AZURE_FOUNDRY_BASE_URL carries for chat/responses."""
    b = (os.environ.get("AZURE_FOUNDRY_BASE_URL") or _env_file_key("AZURE_FOUNDRY_BASE_URL") or _DEFAULT_BASE).rstrip("/")
    for suffix in ("/openai/v1", "/openai", "/v1"):
        if b.endswith(suffix):
            b = b[: -len(suffix)]
            break
    return b


def _model() -> str:
    return os.environ.get(_MODEL_ENV, _DEFAULT_MODEL)


def _call(state: str, questions: Dict[str, Dict[str, Any]], model: Optional[str] = None, timeout: float = 60.0) -> Dict[str, Any]:
    """POST to the System One endpoint and return the parsed response.

    On any transport/HTTP error returns {"error": ...} so callers fail open.
    """
    body = json.dumps({"model": model or _model(), "state": state, "questions": questions}).encode()
    key = _env_file_key("AZURE_FOUNDRY_API_KEY")
    if not key:
        return {"error": "missing AZURE_FOUNDRY_API_KEY"}
    req = urllib.request.Request(
        f"{_base_url()}/providers/microsoft/v1/systemone",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "api-key": key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:  # pragma: no cover - network
        return {"error": f"HTTP {e.code}", "detail": e.read().decode()[:400]}
    except Exception as e:  # pragma: no cover - network
        return {"error": type(e).__name__, "detail": str(e)[:300]}


# ── Tool handlers (each returns a JSON string) ────────────────────────────────

def _noul_handler(args: Mapping[str, Any], **_: Any) -> str:
    state = args.get("state", "")
    question = args.get("question", "")
    resp = _call(state, {"q": {
        "type": "noul",
        "instructions": question,
        "criteria": {"true": args.get("true_label", "true"), "false": args.get("false_label", "false")},
    }})
    return json.dumps(resp)


def _choice_handler(args: Mapping[str, Any], **_: Any) -> str:
    state = args.get("state", "")
    question = args.get("question", "")
    options = args.get("options") or []
    criteria = args.get("criteria") or {o: o for o in options}
    resp = _call(state, {"q": {"type": "choice", "instructions": question, "criteria": criteria}})
    return json.dumps(resp)


def _score_handler(args: Mapping[str, Any], **_: Any) -> str:
    state = args.get("state", "")
    question = args.get("question", "")
    levels = args.get("levels") or []
    resp = _call(state, {"q": {"type": "score", "instructions": question, "criteria": levels}})
    return json.dumps(resp)


def _decide_handler(args: Mapping[str, Any], **_: Any) -> str:
    state = args.get("state", "")
    questions = args.get("questions") or {}
    resp = _call(state, questions)
    return json.dumps(resp)


def check_requirements() -> bool:
    return bool(_env_file_key("AZURE_FOUNDRY_API_KEY"))


def register_tools(ctx) -> None:  # noqa: ANN001
    common = dict(toolset="ms-decision", check_fn=check_requirements, requires_env=["AZURE_FOUNDRY_API_KEY"])
    str_prop = lambda d: {"type": "string", "description": d}  # noqa: E731

    ctx.register_tool(
        name="decision_noul",
        schema={
            "name": "decision_noul",
            "description": "Return the probability (0-1) that a yes/no condition holds for a state. Use for binary classification and gating (is this urgent? safe to run?).",
            "parameters": {"type": "object", "properties": {
                "state": str_prop("The text/state to judge"),
                "question": str_prop("The yes/no question"),
                "true_label": str_prop("Label for the true outcome"),
                "false_label": str_prop("Label for the false outcome"),
            }, "required": ["state", "question"]},
        },
        handler=lambda args, **kw: _noul_handler(args, **kw),
        description="Microsoft-Decision-1 yes/no probability",
        emoji="⚖️",
        **common,
    )
    ctx.register_tool(
        name="decision_choice",
        schema={
            "name": "decision_choice",
            "description": "Pick one option from a fixed set for a state, with per-option probabilities + confidence. Use for routing (which model/team/queue).",
            "parameters": {"type": "object", "properties": {
                "state": str_prop("The text/state to judge"),
                "question": str_prop("The routing/classification question"),
                "options": {"type": "array", "items": {"type": "string"}, "description": "Candidate labels"},
                "criteria": {"type": "object", "description": "Optional option->description map to steer the classifier"},
            }, "required": ["state", "question", "options"]},
        },
        handler=lambda args, **kw: _choice_handler(args, **kw),
        description="Microsoft-Decision-1 fixed-set choice",
        emoji="🧭",
        **common,
    )
    ctx.register_tool(
        name="decision_score",
        schema={
            "name": "decision_score",
            "description": "Score a state on an ordered scale (levels low -> high) with a continuous score + distribution. Use for prioritization/severity.",
            "parameters": {"type": "object", "properties": {
                "state": str_prop("The text/state to judge"),
                "question": str_prop("The scoring question"),
                "levels": {"type": "array", "items": {"type": "string"}, "description": "Ordered levels, low to high"},
            }, "required": ["state", "question", "levels"]},
        },
        handler=lambda args, **kw: _score_handler(args, **kw),
        description="Microsoft-Decision-1 ordered-scale score",
        emoji="📊",
        **common,
    )
    ctx.register_tool(
        name="decide",
        schema={
            "name": "decide",
            "description": "Ask several typed decision questions about a state in one call. questions: {id: {type: noul|choice|score, instructions, criteria}}. Returns answers keyed by id.",
            "parameters": {"type": "object", "properties": {
                "state": str_prop("The text/state to judge"),
                "questions": {"type": "object", "description": "Map of question-id to typed question spec"},
            }, "required": ["state", "questions"]},
        },
        handler=lambda args, **kw: _decide_handler(args, **kw),
        description="Microsoft-Decision-1 batched typed decisions",
        emoji="🧠",
        **common,
    )


# ── pre_tool_call gate ────────────────────────────────────────────────────────

def _gate(tool_name: str, args: Mapping[str, Any], **_: Any) -> Optional[Dict[str, Any]]:
    """Return a block directive when the model is confident the call is unsafe.

    Returns None (allow) otherwise. Fails open on any provider error so a
    down/rate-limited decision endpoint never wedges the agent loop.
    """
    if not _GATE_ENABLED:
        return None
    if tool_name in SAFE_TOOLS:
        return None
    if tool_name not in RISKY_TOOLS and not tool_name.startswith(("mcp__", "browser_")):
        return None

    summary = json.dumps(args)[:600]
    state = f"Tool call: {tool_name} with arguments: {summary}"
    resp = _call(state, {"safe": {
        "type": "noul",
        "instructions": "Is this operation safe to run automatically without user approval?",
        "criteria": {
            "true": "Read-only or clearly reversible, no data loss or shared-state change",
            "false": "Deletes/overwrites data, touches system files, sends messages, spends money, or is hard to reverse",
        },
    }}, timeout=_TIMEOUT)

    try:
        p_safe = resp["answers"]["safe"]["noul"]
    except Exception:
        return None  # fail open
    if p_safe < _BLOCK_THRESHOLD:
        return {
            "action": "block",
            "message": (
                f"[ms-decision-gate] Microsoft-Decision-1 judged this {tool_name} call unsafe to "
                f"auto-run (safety p={p_safe:.2f} < {_BLOCK_THRESHOLD}). Needs explicit user approval."
            ),
        }
    return None


def register_gate(ctx) -> None:  # noqa: ANN001
    ctx.register_hook("pre_tool_call", lambda **kwargs: _gate(
        kwargs.get("tool_name", ""), kwargs.get("args") or kwargs.get("tool_input") or {}, **kwargs
    ))
