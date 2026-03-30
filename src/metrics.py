from __future__ import annotations

from typing import Any, Mapping, Tuple, Dict, List

from src.logger import get_logger

_metrics_log = get_logger("metrics")

# In-process aggregation (lightweight; replace with real backend in prod)
_COUNTERS: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], int] = {}
_HISTOGRAMS: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], List[float]] = {}


def _fmt_labels(labels: Mapping[str, Any] | None) -> str:
    if not labels:
        return ""
    kv = ",".join(f"{k}={v}" for k, v in labels.items())
    return f" {{{kv}}}"

def _labels_key(labels: Mapping[str, Any] | None) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((labels or {}).items()))

def increment_counter(name: str, *, labels: Mapping[str, Any] | None = None, amount: int = 1) -> None:
    """Increment a counter metric (no-op backend by default; logs for observability).

    Replace this with a real metrics backend (Prometheus/OTel) in production.
    """
    _metrics_log.info("counter %s +%s%s", name, amount, _fmt_labels(labels))
    key = (name, _labels_key(labels))
    _COUNTERS[key] = _COUNTERS.get(key, 0) + amount


def observe_histogram(name: str, value: float, *, labels: Mapping[str, Any] | None = None) -> None:
    """Record a value for a histogram metric (logged by default)."""
    _metrics_log.info("histogram %s %s%s", name, value, _fmt_labels(labels))
    key = (name, _labels_key(labels))
    _HISTOGRAMS.setdefault(key, []).append(float(value))


def _percentiles(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    s = sorted(values)
    def perc(p: float) -> float:
        if not s:
            return 0.0
        i = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
        return s[i]
    return perc(50), perc(95)


def get_metrics_summary() -> Dict[str, Any]:
    """Return a snapshot of KPIs computed from in-process aggregates."""
    # Finality
    total_j = sum(v for (n, _), v in _COUNTERS.items() if n == "evaluator_judgements_total")
    accepted_j = sum(
        v
        for (n, labels), v in _COUNTERS.items()
        if n == "evaluator_judgements_total" and dict(labels).get("accepted") == "true"
    )
    finality_pct = (accepted_j / total_j * 100.0) if total_j else 0.0

    # Iterations to success
    iters = []
    for (n, _), vals in _HISTOGRAMS.items():
        if n == "iterations_to_success":
            iters.extend(vals)
    it_p50, it_p95 = _percentiles(iters)

    # E2E latency
    e2e_vals = []
    for (n, _), vals in _HISTOGRAMS.items():
        if n == "e2e_latency_ms":
            e2e_vals.extend(vals)
    e2e_p50, e2e_p95 = _percentiles(e2e_vals)

    # Tokens per run
    tok_vals = []
    for (n, _), vals in _HISTOGRAMS.items():
        if n == "tokens_total":
            tok_vals.extend(vals)
    avg_tokens = (sum(tok_vals) / len(tok_vals)) if tok_vals else 0.0

    # Tool success rates
    per_tool: Dict[str, Dict[str, int]] = {}
    for (n, labels), v in _COUNTERS.items():
        if n == "tool_calls_total":
            d = dict(labels)
            tool = d.get("tool", "unknown")
            status = d.get("status", "unknown")
            per_tool.setdefault(tool, {"success": 0, "error": 0})
            per_tool[tool][status] = per_tool[tool].get(status, 0) + v
    tool_rates: Dict[str, float] = {}
    for tool, counts in per_tool.items():
        total = counts.get("success", 0) + counts.get("error", 0)
        rate = (counts.get("success", 0) / total * 100.0) if total else 0.0
        tool_rates[tool] = rate

    return {
        "finality_pct": finality_pct,
        "iterations_p50": it_p50,
        "iterations_p95": it_p95,
        "e2e_latency_p50_ms": e2e_p50,
        "e2e_latency_p95_ms": e2e_p95,
        "avg_tokens_per_run": avg_tokens,
        "tool_success_rates": tool_rates,
    }


def get_hist_values(name: str) -> List[float]:
    """Return all observed values for a histogram by metric name (merged across labels)."""
    out: List[float] = []
    for (n, _), vals in _HISTOGRAMS.items():
        if n == name:
            out.extend(vals)
    return out


def get_last_value(name: str) -> float | None:
    """Return the most recent observed value for a histogram metric, or None."""
    last: float | None = None
    for (n, _), vals in _HISTOGRAMS.items():
        if n == name and vals:
            v = vals[-1]
            if last is None or v:  # prefer the most recent appended
                last = v
    return last

