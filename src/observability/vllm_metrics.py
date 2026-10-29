"""Scrape inference-server Prometheus ``/metrics`` endpoints.

Focuses on gauges (KV cache, queue depth) and counters (token totals,
preemptions, prefix-cache hits/queries). Counter snapshots are cumulative
since server start; pair two snapshots through :func:`inference_metrics_delta`
to obtain the windowed deltas and hit ratios used for capacity tuning.

Histogram extraction (TTFT, e2e latency) is deferred — those require
maintaining per-request state across scrapes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import requests
from loguru import logger

from src.observability.prometheus import parse_prometheus_text


@dataclass
class InferenceMetricsSnapshot:
    scraped_at: float  # time.time() — for time-series correlation
    kv_cache_usage_pct: Optional[float] = None
    num_requests_running: Optional[int] = None
    num_requests_waiting: Optional[int] = None
    # Counter-based (cumulative since server start; useful via deltas between scrapes)
    total_prompt_tokens: Optional[int] = None
    total_generation_tokens: Optional[int] = None
    # KV-residency / prefix-cache health; inspect as windowed deltas, not raw totals.
    total_preemptions: Optional[int] = None
    prefix_cache_hits: Optional[int] = None
    prefix_cache_queries: Optional[int] = None
    external_prefix_cache_hits: Optional[int] = None
    external_prefix_cache_queries: Optional[int] = None
    prompt_tokens_cached: Optional[int] = None


VllmMetricsSnapshot = InferenceMetricsSnapshot


@dataclass
class InferenceMetricsDelta:
    """Windowed deltas between two :class:`InferenceMetricsSnapshot` scrapes.

    Counter fields are ``curr - prev`` over the window. A counter that went
    backwards (server restart) or is missing on either side yields ``None``.
    Ratios are computed from the *window* deltas, not cumulative totals.
    Gauges reflect the later (``curr``) snapshot.
    """

    window_seconds: float
    preemptions: Optional[int] = None
    prompt_tokens: Optional[int] = None
    generation_tokens: Optional[int] = None
    prefix_cache_hits: Optional[int] = None
    prefix_cache_queries: Optional[int] = None
    prefix_cache_hit_rate: Optional[float] = None
    external_prefix_cache_hits: Optional[int] = None
    external_prefix_cache_queries: Optional[int] = None
    external_prefix_cache_hit_rate: Optional[float] = None
    prompt_tokens_cached: Optional[int] = None
    prompt_tokens_cached_rate: Optional[float] = None
    prompt_tokens_per_second: Optional[float] = None
    generation_tokens_per_second: Optional[float] = None
    # Gauges (instantaneous, from the later snapshot)
    kv_cache_usage_pct: Optional[float] = None
    num_requests_running: Optional[int] = None
    num_requests_waiting: Optional[int] = None


# Per backend: snapshot field -> ordered candidate Prometheus metric names.
# Counters list ``_total`` first (the form the pinned vLLM nightly emits) then
# the bare name (current vLLM docs). Exact-name lookup means the ``_created``
# unix-timestamp siblings are never matched.
_METRIC_NAMES: dict[str, dict[str, tuple[str, ...]]] = {
    "vllm": {
        "kv_cache_usage": ("vllm:kv_cache_usage_perc",),
        "num_requests_running": ("vllm:num_requests_running",),
        "num_requests_waiting": ("vllm:num_requests_waiting",),
        "total_prompt_tokens": ("vllm:prompt_tokens_total", "vllm:prompt_tokens"),
        "total_generation_tokens": (
            "vllm:generation_tokens_total",
            "vllm:generation_tokens",
        ),
        "total_preemptions": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
        "prefix_cache_hits": (
            "vllm:prefix_cache_hits_total",
            "vllm:prefix_cache_hits",
        ),
        "prefix_cache_queries": (
            "vllm:prefix_cache_queries_total",
            "vllm:prefix_cache_queries",
        ),
        "external_prefix_cache_hits": (
            "vllm:external_prefix_cache_hits_total",
            "vllm:external_prefix_cache_hits",
        ),
        "external_prefix_cache_queries": (
            "vllm:external_prefix_cache_queries_total",
            "vllm:external_prefix_cache_queries",
        ),
        "prompt_tokens_cached": (
            "vllm:prompt_tokens_cached_total",
            "vllm:prompt_tokens_cached",
        ),
    },
    "sglang": {
        "kv_cache_usage": ("sglang:token_usage",),
        "num_requests_running": ("sglang:num_running_reqs",),
        "num_requests_waiting": ("sglang:num_queue_reqs",),
        "total_prompt_tokens": ("sglang:prompt_tokens_total",),
        "total_generation_tokens": ("sglang:gen_throughput_total",),
    },
}


def snapshot_inference_metrics(
    metrics_url: str,
    *,
    backend: Literal["vllm", "sglang"] = "vllm",
    timeout: float = 2.0,
    api_key: str | None = None,
) -> Optional[InferenceMetricsSnapshot]:
    """Scrape an inference ``/metrics`` endpoint, returning ``None`` on failure."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(metrics_url, timeout=timeout, headers=headers)
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403):
            logger.warning(
                f"{backend} /metrics returned {status} — check API key configuration"
            )
        else:
            logger.debug(f"{backend} /metrics scrape failed: {exc}")
        return None
    except Exception as exc:
        logger.debug(f"{backend} /metrics scrape failed: {exc}")
        return None

    parsed = parse_prometheus_text(response.text)
    scraped_at = time.time()
    names = _METRIC_NAMES[backend]

    def gauge_pct(field: str) -> Optional[float]:
        return _first_pct(parsed, names.get(field, ()))

    def gauge_int(field: str) -> Optional[int]:
        return _first_int(parsed, names.get(field, ()))

    return InferenceMetricsSnapshot(
        scraped_at=scraped_at,
        kv_cache_usage_pct=gauge_pct("kv_cache_usage"),
        num_requests_running=gauge_int("num_requests_running"),
        num_requests_waiting=gauge_int("num_requests_waiting"),
        total_prompt_tokens=gauge_int("total_prompt_tokens"),
        total_generation_tokens=gauge_int("total_generation_tokens"),
        total_preemptions=gauge_int("total_preemptions"),
        prefix_cache_hits=gauge_int("prefix_cache_hits"),
        prefix_cache_queries=gauge_int("prefix_cache_queries"),
        external_prefix_cache_hits=gauge_int("external_prefix_cache_hits"),
        external_prefix_cache_queries=gauge_int("external_prefix_cache_queries"),
        prompt_tokens_cached=gauge_int("prompt_tokens_cached"),
    )


def snapshot_vllm_metrics(
    metrics_url: str,
    timeout: float = 2.0,
    api_key: str | None = None,
) -> Optional[VllmMetricsSnapshot]:
    """Backward-compatible vLLM metrics scraper wrapper."""

    return snapshot_inference_metrics(
        metrics_url,
        backend="vllm",
        timeout=timeout,
        api_key=api_key,
    )


def inference_metrics_delta(
    prev: InferenceMetricsSnapshot,
    curr: InferenceMetricsSnapshot,
) -> InferenceMetricsDelta:
    """Compute windowed deltas + hit ratios between two snapshots.

    ``prev`` is the earlier scrape, ``curr`` the later one. Designed so the
    caller can sample the live server twice ``N`` seconds apart and read off
    preemption rate, prefix-cache hit ratio, and token throughput for that
    window without conflating it with cold-start cumulative totals.
    """
    window = curr.scraped_at - prev.scraped_at
    if window < 0:
        window = 0.0

    def counter_delta(a: Optional[int], b: Optional[int]) -> Optional[int]:
        if a is None or b is None:
            return None
        diff = b - a
        # Negative => counter reset (server restart); not a trustworthy delta.
        return diff if diff >= 0 else None

    def ratio(num: Optional[int], den: Optional[int]) -> Optional[float]:
        if num is None or den is None or den <= 0:
            return None
        return num / den

    def per_second(d: Optional[int]) -> Optional[float]:
        if d is None or window <= 0:
            return None
        return d / window

    preemptions = counter_delta(prev.total_preemptions, curr.total_preemptions)
    prompt_tokens = counter_delta(prev.total_prompt_tokens, curr.total_prompt_tokens)
    generation_tokens = counter_delta(
        prev.total_generation_tokens, curr.total_generation_tokens
    )
    pc_hits = counter_delta(prev.prefix_cache_hits, curr.prefix_cache_hits)
    pc_queries = counter_delta(prev.prefix_cache_queries, curr.prefix_cache_queries)
    ext_hits = counter_delta(
        prev.external_prefix_cache_hits, curr.external_prefix_cache_hits
    )
    ext_queries = counter_delta(
        prev.external_prefix_cache_queries, curr.external_prefix_cache_queries
    )
    cached = counter_delta(prev.prompt_tokens_cached, curr.prompt_tokens_cached)

    return InferenceMetricsDelta(
        window_seconds=window,
        preemptions=preemptions,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        prefix_cache_hits=pc_hits,
        prefix_cache_queries=pc_queries,
        prefix_cache_hit_rate=ratio(pc_hits, pc_queries),
        external_prefix_cache_hits=ext_hits,
        external_prefix_cache_queries=ext_queries,
        external_prefix_cache_hit_rate=ratio(ext_hits, ext_queries),
        prompt_tokens_cached=cached,
        prompt_tokens_cached_rate=ratio(cached, prompt_tokens),
        prompt_tokens_per_second=per_second(prompt_tokens),
        generation_tokens_per_second=per_second(generation_tokens),
        kv_cache_usage_pct=curr.kv_cache_usage_pct,
        num_requests_running=curr.num_requests_running,
        num_requests_waiting=curr.num_requests_waiting,
    )


def _first_value(parsed: dict, candidates: Sequence[str]) -> Optional[float]:
    """First value for the first candidate metric name present in ``parsed``."""
    for name in candidates:
        entries = parsed.get(name)
        if entries:
            _, value = entries[0]
            return value
    return None


def _first_pct(parsed: dict, candidates: Sequence[str]) -> Optional[float]:
    """Extract a gauge value and convert from fraction to percent."""
    value = _first_value(parsed, candidates)
    return None if value is None else value * 100.0


def _first_int(parsed: dict, candidates: Sequence[str]) -> Optional[int]:
    """Extract a value as an integer."""
    value = _first_value(parsed, candidates)
    return None if value is None else int(value)
