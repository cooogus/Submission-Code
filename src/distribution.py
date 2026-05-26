#!/usr/bin/env python3
"""Pass@4 gap distribution and CDF"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

GAP_THRESHOLDS_PP = (2.0, 5.0, 10.0)
PERCENTILES = (25, 50, 75, 90, 95, 100)


def pass4_gaps_pp(pairs: list[dict[str, Any]]) -> list[float]:
    return [float(row["pass4_gap"]) * 100.0 for row in pairs]


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _summary_stats(gaps_pp: list[float]) -> dict[str, float]:
    if not gaps_pp:
        return {}
    ordered = sorted(gaps_pp)
    n = len(ordered)
    return {
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "mean": round(sum(ordered) / n, 4),
        "median": round(_percentile(ordered, 50), 4),
        **{
            f"p{int(p)}": round(_percentile(ordered, p), 4) for p in PERCENTILES if p < 100
        },
        "p100": round(ordered[-1], 4),
    }


def _threshold_block(gaps_pp: list[float], thresholds_pp: tuple[float, ...] = GAP_THRESHOLDS_PP) -> dict[str, Any]:
    n = len(gaps_pp)
    count_above: dict[str, int] = {}
    fraction_above: dict[str, float] = {}
    percent_above: dict[str, float] = {}
    for threshold in thresholds_pp:
        key = str(int(threshold) if threshold == int(threshold) else threshold)
        count = sum(1 for gap in gaps_pp if gap > threshold)
        frac = count / n if n else 0.0
        count_above[key] = count
        fraction_above[key] = round(frac, 6)
        percent_above[key] = round(frac * 100.0, 2)
    return {
        "thresholds_pp": list(thresholds_pp),
        "count_above": count_above,
        "fraction_above": fraction_above,
        "percent_above": percent_above,
    }


def distribution_for_pairs(
    pairs: list[dict[str, Any]],
    *,
    pair_count: int | None = None,
) -> dict[str, Any]:
    gaps_pp = pass4_gaps_pp(pairs)
    n = pair_count if pair_count is not None else len(gaps_pp)
    if n != len(gaps_pp):
        raise ValueError(f"pair_count {n} != len(pairs) {len(gaps_pp)}")

    by_benchmark: dict[str, list[float]] = defaultdict(list)
    for pair, gap_pp in zip(pairs, gaps_pp):
        by_benchmark[str(pair["benchmark_label"])].append(gap_pp)

    return {
        "pair_count": n,
        "gap_unit": "percentage_points",
        "summary_pp": _summary_stats(gaps_pp),
        "thresholds": _threshold_block(gaps_pp),
        "by_benchmark": {
            bench: {
                "pair_count": len(bench_gaps),
                "summary_pp": _summary_stats(bench_gaps),
                "thresholds": _threshold_block(bench_gaps),
            }
            for bench, bench_gaps in sorted(by_benchmark.items())
        },
    }


def paper_ready_sentence(distribution: dict[str, Any]) -> str:
    n = distribution["pair_count"]
    pct = distribution["thresholds"]["percent_above"]
    median = distribution["summary_pp"]["median"]
    maximum = distribution["summary_pp"]["max"]
    return (
        f"Across {n} qualifying matched-Pass@1 pairs, "
        f"{pct['2']:.1f}% show Pass@4 divergence above 2 percentage points, "
        f"{pct['5']:.1f}% above 5 points, and {pct['10']:.1f}% above 10 points "
        f"(median gap {median:.2f} pp; maximum {maximum:.2f} pp)."
    )


def build_cdf_rows(gaps_pp: list[float]) -> list[dict[str, float | int]]:
    if not gaps_pp:
        return []
    ordered = sorted(gaps_pp)
    n = len(ordered)
    counts = Counter(ordered)
    unique = sorted(counts)
    cumulative = 0
    rows: list[dict[str, float | int]] = []
    for gap_pp in unique:
        cumulative += counts[gap_pp]
        rows.append(
            {
                "gap_pp": round(gap_pp, 6),
                "n_pairs_at_or_below": cumulative,
                "fraction_at_or_below": round(cumulative / n, 6),
                "n_pairs_total": n,
            }
        )
    return rows


def write_cdf_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_cdf_rows(pass4_gaps_pp(pairs))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gap_pp",
                "n_pairs_at_or_below",
                "fraction_at_or_below",
                "n_pairs_total",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_cdf_figure(path: Path, pairs: list[dict[str, Any]], *, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    gaps_pp = pass4_gaps_pp(pairs)
    if not gaps_pp:
        return False

    ordered = sorted(gaps_pp)
    n = len(ordered)
    x = sorted(set(ordered))
    y = [sum(1 for g in ordered if g <= xp) / n for xp in x]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.step(x, y, where="post", color="#1f4e79", linewidth=2)
    for threshold in GAP_THRESHOLDS_PP:
        ax.axvline(threshold, color="#888888", linestyle="--", linewidth=0.9, alpha=0.8)
        frac = sum(1 for g in gaps_pp if g > threshold) / n
        ax.text(
            threshold,
            0.02,
            f">{threshold} pp: {100 * frac:.0f}%",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="#444444",
        )
    ax.set_xlim(0, max(x) * 1.05 if x else 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Pass@4 gap (percentage points)")
    ax.set_ylabel("Fraction of pairs at or below gap")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_distribution_artifacts(
    out_dir: Path,
    scope: str,
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    distribution = distribution_for_pairs(pairs)
    distribution["paper_ready_sentence"] = paper_ready_sentence(distribution)

    prefix = f"{scope}_pass4_gap"
    dist_json = out_dir / f"{prefix}_distribution.json"
    dist_json.write_text(json.dumps(distribution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_cdf_csv(out_dir / f"{prefix}_cdf.csv", pairs)
    figure_path = out_dir / f"{prefix}_cdf.png"
    figure_written = write_cdf_figure(
        figure_path,
        pairs,
        title=f"P2 Pass@4 gap CDF ({scope} scope, n={len(pairs)})",
    )

    return {
        "distribution_json": str(dist_json.resolve()),
        "cdf_csv": str((out_dir / f"{prefix}_cdf.csv").resolve()),
        "cdf_png": str(figure_path.resolve()) if figure_written else None,
        "distribution": distribution,
    }
