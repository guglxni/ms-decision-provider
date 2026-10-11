"""Microsoft-Decision-1 (System One) decision provider for Hermes.

Directory plugin entry point. `register(ctx)` wires the typed-decision tools
into the tool registry and (optionally) a `pre_tool_call` hook that consults
Microsoft-Decision-1 before risky tool calls.

The provider preserves full probability distributions; the *consumer* owns the
policy threshold (see the gate's BLOCK_THRESHOLD). This mirrors the contract
proposed in NousResearch/hermes-agent#113008 — bounded semantic questions whose
output space is known in advance and whose useful output is a probability
distribution, not prose.
"""
from __future__ import annotations

__version__ = "1.0.0"


def register(ctx):  # noqa: ANN001 - plugin context is duck-typed by Hermes
    try:
        from . import provider  # package-qualified load (Hermes directory plugin)
    except ImportError:
        import provider  # direct/script load fallback

    provider.register_tools(ctx)
    provider.register_gate(ctx)
