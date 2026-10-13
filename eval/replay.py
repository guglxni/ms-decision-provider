#!/usr/bin/env python3
"""Shadow replay harness for the ms-decision-provider plugin.

Replays a labeled decision corpus through Microsoft-Decision-1 (System One)
WITHOUT changing any Hermes behavior, then scores each decision family:

  noul   -> top-1 accuracy, Brier score, calibration-by-bucket, risk-vs-coverage
  choice -> top-1 accuracy, mean confidence, coverage at threshold
  score  -> exact-level accuracy, adjacent (off-by-one) accuracy, MAE

This implements the shadow-mode / replay acceptance criteria from
NousResearch/hermes-agent#113008: evaluate decision *families*, not models
globally; abstention is first-class; thresholds are owned by the family.

Usage:
    python3 eval/replay.py                          # all families, real provider
    python3 eval/replay.py --family tool_safety     # one family
    python3 eval/replay.py --coverage 0.7           # abstain below p=0.7
    python3 eval/replay.py --cases my_cases.json    # custom corpus

Requires AZURE_FOUNDRY_API_KEY (resolved via Hermes's secret scope, or env).
Fails closed with a clear message if the provider is unreachable — a replay
harness must not silently fabricate results.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

_MODEL = os.environ.get("MS_DECISION_MODEL", "ms-decision-1")
_TIMEOUT = float(os.environ.get("MS_DECISION_TIMEOUT", "60"))


def _get_secret(name: str) -> str:
    """Resolve via Hermes's public secret scope, env fallback, never .env."""
    try:
        from agent.secret_scope import get_secret  # type: ignore

        val = get_secret(name)
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(name, "")


def _base_url() -> str:
    b = _get_secret("AZURE_FOUNDRY_BASE_URL").rstrip("/")
    if not b:
        raise RuntimeError("set AZURE_FOUNDRY_BASE_URL to your own Foundry resource")
    for suffix in ("/openai/v1", "/openai", "/v1"):
        if b.endswith(suffix):
            b = b[: -len(suffix)]
            break
    return b


def _call(state: str, questions: dict) -> dict:
    """One System One call. Raises on any failure (replay must not fabricate)."""
    body = json.dumps({"model": _MODEL, "state": state, "questions": questions}).encode()
    key = _get_secret("AZURE_FOUNDRY_API_KEY")
    if not key:
        raise RuntimeError("AZURE_FOUNDRY_API_KEY not set (set it in your secret scope or environment)")
    req = urllib.request.Request(
        f"{_base_url()}/providers/microsoft/v1/systemone",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}", "api-key": key},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode())


# ── scoring helpers ───────────────────────────────────────────────────────────

def _brier(p_true: float, label: bool) -> float:
    return (p_true - (1.0 if label else 0.0)) ** 2


def _bucket_calibration(rows: list) -> dict:
    """rows: list of (p_true, label_bool). Buckets of 0.1."""
    buckets: dict = {}
    for p, lab in rows:
        b = min(9, int(p * 10))
        buckets.setdefault(b, []).append((p, lab))
    out = {}
    for b, rs in sorted(buckets.items()):
        mean_p = sum(r[0] for r in rs) / len(rs)
        frac_true = sum(1 for r in rs if r[1]) / len(rs)
        out[f"{b / 10:.1f}-{b / 10 + 0.1:.1f}"] = {"n": len(rs), "mean_pred": round(mean_p, 3), "frac_true": round(frac_true, 3)}
    return out


def replay_family(family: dict, coverage: float, delay: float) -> dict:
    qtype = family["question_type"]
    results = []
    for case in family["cases"]:
        q = {"type": qtype, "instructions": family["instructions"], "criteria": family["criteria"]}
        t = time.time()
        resp = _call(case["state"], {"q": q})
        latency_ms = int((time.time() - t) * 1000)
        if "error" in resp:
            raise RuntimeError(f"provider error on {case['id']}: {resp}")
        ans = resp["answers"]["q"]
        usage = resp.get("usage", {})
        results.append({"id": case["id"], "label": case["label"], "answer": ans,
                        "latency_ms": latency_ms, "usage": usage})
        if delay:
            time.sleep(delay)

    metrics: dict = {"n": len(results), "coverage_threshold": coverage}
    latencies = [r["latency_ms"] for r in results]
    metrics["latency_ms"] = {"p50": sorted(latencies)[len(latencies) // 2],
                             "max": max(latencies),
                             "mean": round(sum(latencies) / len(latencies), 1)}
    metrics["tokens"] = {"input": sum(r["usage"].get("input_tokens", 0) for r in results),
                         "output": sum(r["usage"].get("output_tokens", 0) for r in results)}

    if qtype == "noul":
        rows = [(r["answer"]["noul"], bool(r["label"])) for r in results]
        correct = sum(1 for p, lab in rows if (p >= 0.5) == lab)
        metrics["accuracy"] = round(correct / len(rows), 3)
        metrics["brier"] = round(sum(_brier(p, lab) for p, lab in rows) / len(rows), 4)
        metrics["calibration"] = _bucket_calibration(rows)
        # risk-vs-coverage: of cases where the model is confident enough to act
        # (dist from 0.5 >= coverage margin), what fraction are correct?
        confident = [(p, lab) for p, lab in rows if abs(p - 0.5) >= (coverage - 0.5) * 2 or abs(p - 0.5) >= (1 - coverage)]
        confident = [(p, lab) for p, lab in rows if max(p, 1 - p) >= coverage]
        if confident:
            cc = sum(1 for p, lab in confident if (p >= 0.5) == lab)
            metrics["risk_vs_coverage"] = {"coverage": round(len(confident) / len(rows), 3),
                                           "accuracy_when_confident": round(cc / len(confident), 3)}
    elif qtype == "choice":
        correct = sum(1 for r in results if r["answer"]["choice"] == r["label"])
        metrics["accuracy"] = round(correct / len(results), 3)
        confs = [r["answer"].get("confidence", 0.0) for r in results]
        metrics["mean_confidence"] = round(sum(confs) / len(confs), 3)
        confident = [r for r in results if r["answer"].get("confidence", 0.0) >= coverage]
        if confident:
            cc = sum(1 for r in confident if r["answer"]["choice"] == r["label"])
            metrics["risk_vs_coverage"] = {"coverage": round(len(confident) / len(results), 3),
                                           "accuracy_when_confident": round(cc / len(confident), 3)}
        metrics["per_case"] = [{"id": r["id"], "label": r["label"],
                                "picked": r["answer"]["choice"],
                                "confidence": round(r["answer"].get("confidence", 0.0), 3),
                                "correct": r["answer"]["choice"] == r["label"]} for r in results]
    elif qtype == "score":
        # legend maps index->level name; score is continuous on 0..len(levels)-1
        exact = 0
        adjacent = 0
        abs_err = []
        for r in results:
            pred = r["answer"]["score"]
            lab = float(r["label"])
            abs_err.append(abs(pred - lab))
            if round(pred) == lab:
                exact += 1
            if abs(round(pred) - lab) <= 1:
                adjacent += 1
        n = len(results)
        metrics["exact_level_accuracy"] = round(exact / n, 3)
        metrics["adjacent_accuracy"] = round(adjacent / n, 3)
        metrics["mae"] = round(sum(abs_err) / n, 3)
        metrics["per_case"] = [{"id": r["id"], "label": r["label"],
                                "score": round(r["answer"]["score"], 2)} for r in results]

    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay labeled decisions through Microsoft-Decision-1")
    ap.add_argument("--cases", default=os.path.join(os.path.dirname(__file__), "decision_cases.json"))
    ap.add_argument("--family", help="only replay one family by name")
    ap.add_argument("--coverage", type=float, default=0.7, help="confidence threshold for risk-vs-coverage (default 0.7)")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between calls (rate-limit friendly)")
    args = ap.parse_args()

    corpus = json.load(open(args.cases))
    families = corpus["families"]
    if args.family:
        families = [f for f in families if f["name"] == args.family]
        if not families:
            print(f"no family named {args.family!r}", file=sys.stderr)
            return 2

    print(f"# ms-decision replay — model={_MODEL} base={_base_url()}\n")
    report = {}
    for fam in families:
        print(f"▸ {fam['name']} ({fam['question_type']}, {len(fam['cases'])} cases)…", flush=True)
        report[fam["name"]] = replay_family(fam, args.coverage, args.delay)
        print(f"  done\n")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
