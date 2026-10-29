"""Lightweight parser for Prometheus text exposition format.

Parses gauge, counter, histogram bucket/sum/count lines.  Skips ``# HELP``
and ``# TYPE`` comment lines.  No external dependency needed — the format is
simple enough for line-by-line parsing.
"""

from __future__ import annotations

import re


# Matches: metric_name{label="value",...} numeric_value
# or:      metric_name numeric_value  (no labels)
_METRIC_LINE_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{(?P<labels>[^}]*)\})?\s+'
    r'(?P<value>[^\s]+)$'
)

_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_prometheus_text(
    text: str,
) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse Prometheus exposition format into ``{metric_name: [(labels, value), ...]}``.

    Each metric name maps to a list of (labels_dict, float_value) tuples,
    since a single metric name can appear multiple times with different label
    combinations (e.g. histogram buckets, multi-instance gauges).
    """
    metrics: dict[str, list[tuple[dict[str, str], float]]] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        match = _METRIC_LINE_RE.match(line)
        if match is None:
            continue

        name = match.group("name")
        labels_str = match.group("labels") or ""
        raw_value = match.group("value")

        try:
            value = float(raw_value)
        except ValueError:
            continue

        labels = dict(_LABEL_RE.findall(labels_str))

        metrics.setdefault(name, []).append((labels, value))

    return metrics
