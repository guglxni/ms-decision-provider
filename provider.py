"""Microsoft-Decision-1 client + Hermes tool/hook registration.

Calls Azure AI Foundry's System One endpoint:
    POST {base}/providers/microsoft/v1/systemone

Typed questions:
    noul   -> yes/no probability           {"noul": 0.93}
    choice -> pick from a fixed set        {"choice": "infra", "probabilities": {...}}
    score  -> ordered-scale score          {"score": 2.94, "legend": {...}, "probabilities": {...}}

Every handler returns a JSON string and preserves the full distribution so the
caller can apply its own threshold / abstention policy.

Credentials and the endpoint are resolved at call time through Hermes's public
secret route (``agent.secret_scope.get_secret``) — this plugin never opens
``~/.hermes/.env`` itself and never sends anything anywhere unless
``AZURE_FOUNDRY_BASE_URL`` is explicitly set.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import shlex
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Mapping, Optional

_MODEL_ENV = "MS_DECISION_MODEL"
_DEFAULT_MODEL = "ms-decision-1"

# How the gate escalates: "block" hard-denies the call; "approve" hands it to
# Hermes's human approval prompt (matches the "needs explicit approval" text).
_GATE_MODE = os.environ.get("MS_DECISION_GATE_MODE", "approve")

# Opt-in advisory gate: narrow set of known tools. Tools not listed here
# follow Hermes's native guardrails; this plugin is not a universal sandbox.
RISKY_TOOLS = frozenset({
    "terminal", "execute_code", "write_file", "patch", "process_manage",
    "browser_navigate", "browser_click", "browser_type", "browser_exec",
    "computer_use", "cronjob_manage", "delegate_task", "send_message",
    "skill_manage",
})
# Decision tools expose an explicit state string to Azure by design; do not
# apply the safety gate to their own calls (would recurse).
DECISION_TOOLS = frozenset({
    "decide", "decision_noul", "decision_choice", "decision_score",
    "mcp__ms_decision__decide", "mcp__ms_decision__decision_noul",
    "mcp__ms_decision__decision_choice", "mcp__ms_decision__decision_score",
})

# Gate policy knobs (env-overridable). The consumer owns the threshold.
_BLOCK_THRESHOLD = float(os.environ.get("MS_DECISION_BLOCK_THRESHOLD", "0.5"))
_TIMEOUT = float(os.environ.get("MS_DECISION_TIMEOUT", "20"))
_GATE_ENABLED = os.environ.get("MS_DECISION_GATE", "0") in ("1", "true", "yes")
# A random per-process HMAC key prevents persistent approval rule keys from
# becoming a deterministic hash oracle for private tool arguments.
_RULE_KEY_SECRET = secrets.token_bytes(32)


def _get_secret(name: str) -> str:
    """Resolve at call time using Hermes's profile-aware credential scope.

    A missing or unscoped value must never fall back to another profile's
    process environment. In particular, do not catch UnscopedSecretError.
    """
    from agent.secret_scope import get_secret

    return get_secret(name) or ""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so auth headers are never re-sent to a second host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def _base_url() -> str:
    """Accept only explicit HTTPS Azure OpenAI resource hosts.

    System One hangs off the host root, not the chat API's /openai/v1.
    Reject arbitrary domains, userinfo, ports, query strings, and extra paths
    so a misconfigured endpoint cannot receive a credential or state text.
    """
    raw = _get_secret("AZURE_FOUNDRY_BASE_URL").strip()
    if not raw:
        return ""
    try:
        parsed = urllib.parse.urlsplit(raw)
        host = (parsed.hostname or "").lower()
        if (parsed.scheme != "https" or not host.endswith(".openai.azure.com")
                or not re.fullmatch(r"[a-z0-9-]+\.openai\.azure\.com", host)
                or parsed.username is not None or parsed.password is not None
                or parsed.port is not None
                or parsed.query or parsed.fragment
                or parsed.path.rstrip("/") not in ("", "/openai", "/openai/v1")):
            return ""
        return f"https://{host}"
    except ValueError:
        return ""


def _model() -> str:
    return os.environ.get(_MODEL_ENV, _DEFAULT_MODEL)


def _call(state: str, questions: Dict[str, Dict[str, Any]], model: Optional[str] = None, timeout: float = 60.0) -> Dict[str, Any]:
    """POST to the System One endpoint and return the parsed response.

    Sends nothing when the endpoint/key are missing or unscoped. Transport and
    HTTP errors return sanitized {"error": ...} without body or URL text.
    """
    try:
        base = _base_url()
        key = _get_secret("AZURE_FOUNDRY_API_KEY") if base else ""
    except Exception:
        # UnscopedSecretError in a multiplexed gateway is fail-closed:
        # never read the process environment as a substitute.
        return {"error": "credential scope unavailable"}
    if not base:
        return {"error": "set AZURE_FOUNDRY_BASE_URL"}
    if not key:
        return {"error": "missing AZURE_FOUNDRY_API_KEY"}
    body = json.dumps({"model": model or _model(), "state": state, "questions": questions}).encode()
    # Azure System One accepts the Foundry API key via api-key. Do not send
    # duplicate credential-bearing headers when one is sufficient.
    req = urllib.request.Request(
        f"{base}/providers/microsoft/v1/systemone",
        data=body,
        headers={"Content-Type": "application/json", "api-key": key},
    )
    try:
        with _opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:  # pragma: no cover - network
        e.close()
        return {"error": f"HTTP {e.code}"}  # Never surface an echoed state/key.
    except Exception as e:  # pragma: no cover - network
        return {"error": type(e).__name__}  # Exception text may contain URL/PII.


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
    """Tools surface only when both the key and endpoint are scoped/configured."""
    try:
        return bool(_get_secret("AZURE_FOUNDRY_API_KEY")) and bool(_base_url())
    except Exception:
        return False


def register_tools(ctx) -> None:  # noqa: ANN001
    common = dict(toolset="ms-decision", check_fn=check_requirements,
                  requires_env=["AZURE_FOUNDRY_API_KEY", "AZURE_FOUNDRY_BASE_URL"])
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

def _project_operation(tool_name: str, args: Mapping[str, Any]) -> str:
    """Transmit only tool/operation identifiers, never arguments, paths or PII.

    This is intentionally coarse: a classifier cannot safely inspect arbitrary
    tool contents without an explicit data-egress contract. For ambiguous calls
    we require human approval locally instead of transmitting private content.
    """
    operation = ""
    if tool_name == "terminal":
        try:
            words = shlex.split(str(args.get("command", "")))
            allowed = {"rm", "git", "ls", "mkdir", "curl", "python", "python3",
                       "brew", "npm", "pip", "mv", "cp", "cat", "chmod", "sudo"}
            if words and words[0] in allowed:
                operation = words[0]
                if words[0] == "git" and len(words) > 1 and words[1] in {
                    "push", "reset", "clean", "status", "diff", "log", "checkout", "commit"
                }:
                    operation += " " + words[1]
        except (TypeError, ValueError):
            pass
    elif tool_name in {"cronjob_manage", "process_manage"}:
        candidate = args.get("action", "")
        if isinstance(candidate, str) and candidate in {
            "create", "update", "remove", "pause", "resume", "run", "kill", "poll", "list"
        }:
            operation = candidate
    # MCP/browser tool names can come from third-party integrations. Reduce
    # them to a fixed category so even a malicious name cannot enter egress.
    if tool_name.startswith("mcp__"):
        label = "mcp_tool"
    elif tool_name.startswith("browser_"):
        label = "browser_tool"
    else:
        label = tool_name if tool_name in RISKY_TOOLS else "other_tool"
    return f"Tool call: {label}; operation: {operation or 'unspecified'}"


def _local_approval_summary(tool_name: str, args: Mapping[str, Any]) -> str:
    """Show the complete proposed call to the human, never the remote model.

    Hermes's secret redactor is forced on for this boundary even when global
    output redaction is disabled. The user must see the *whole* action; a
    truncated command could hide a destructive suffix. This description is
    consumed by the local approval UI, not included in the Azure request.
    """
    from agent.redact import redact_sensitive_text

    raw = json.dumps(args, ensure_ascii=True, sort_keys=True)
    safe = redact_sensitive_text(raw, force=True, redact_url_credentials=True)
    return f"{tool_name} arguments:\n{safe}"


def _gate(tool_name: str, args: Mapping[str, Any], **_: Any) -> Optional[Dict[str, Any]]:
    """Escalate known-risk operations to Hermes's human approval gate.

    Only allowlisted operation names, never raw arguments, are sent to Azure.
    The score is advisory; errors and malformed responses do not imply allow.
    Human approval is host-enforced and subject to Hermes's own bypass modes.
    """
    if not _GATE_ENABLED:
        return None
    if tool_name in DECISION_TOOLS:
        return None
    if tool_name not in RISKY_TOOLS and not tool_name.startswith(("mcp__", "browser_")):
        return None
    if not isinstance(args, Mapping):
        args = {}

    try:
        state = _project_operation(tool_name, args)
    except (TypeError, ValueError):
        state = "Tool call: other_tool; operation: unspecified"
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
        if not isinstance(p_safe, (int, float)) or isinstance(p_safe, bool) or not 0 <= p_safe <= 1:
            raise ValueError("invalid probability")
    except (TypeError, KeyError, ValueError):
        p_safe = None  # unavailable / malformed is ambiguous, not permission to bypass

    if _GATE_MODE not in ("approve", "block"):
        return {"action": "block", "message": "[ms-decision-gate] invalid gate mode"}
    # The classifier sees no raw arguments, so it cannot prove an operation
    # safe. Every tool in the explicit risk set requires human approval;
    # the remote score is advisory and may additionally hard-block in block mode.
    if p_safe is None or p_safe < _BLOCK_THRESHOLD:
        message = (
            f"[ms-decision-gate] {tool_name}: decision unavailable or below "
            "the configured safety threshold; human approval required."
        )
    else:
        message = (
            f"[ms-decision-gate] {tool_name}: model signal is advisory "
            "(arguments stayed local); human approval required."
        )
    # The full proposed action is shown only to the local human approval UI.
    # Never place it in the Azure state or a persistent allowlist key.
    try:
        message += "\n" + _local_approval_summary(tool_name, args)
    except Exception:
        return {"action": "block", "message": "[ms-decision-gate] cannot display proposed action for approval"}
    if _GATE_MODE == "block" and (p_safe is None or p_safe < _BLOCK_THRESHOLD):
        return {"action": "block", "message": message}
    # Approval caches must be specific to the full call, not just the tool
    # name; an HMAC prevents the on-disk key from revealing simple arguments.
    try:
        serialized_args = json.dumps([tool_name, args], sort_keys=True, default=str).encode()
    except (TypeError, ValueError, RecursionError):
        # No stable per-call identity -> do not permit an approval cache hit.
        return {"action": "block", "message": "[ms-decision-gate] cannot fingerprint tool arguments"}
    rule_key = "ms-decision:" + hmac.new(
        _RULE_KEY_SECRET, serialized_args, hashlib.sha256,
    ).hexdigest()
    return {"action": "approve", "message": message, "rule_key": rule_key}


def register_gate(ctx) -> None:  # noqa: ANN001
    def gate_hook(**kwargs):
        return _gate(kwargs.get("tool_name", ""), kwargs.get("args") or {})

    ctx.register_hook("pre_tool_call", gate_hook)
