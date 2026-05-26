#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PAIR_COLUMNS = [
    "benchmark_label",
    "left_model",
    "right_model",
    "left_display_name",
    "right_display_name",
    "alpha_gap",
    "beta_gap",
    "gamma_gap",
    "pass1_gap",
    "pass4_gap",
    "left_alpha",
    "right_alpha",
    "left_beta",
    "right_beta",
    "left_gamma",
    "right_gamma",
    "left_pass4",
    "right_pass4",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return Path(__file__).resolve().parent


def source_path(scope: str) -> Path:
    source_dir = repo_root() / "P3" / "artifacts"
    if scope == "group":
        return source_dir / "psf_group.json"
    if scope == "variant":
        return source_dir / "psf_variant.json"
    raise ValueError(f"unknown scope: {scope}")


def output_dir() -> Path:
    return project_root() / "p2_outputs"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected a list in {path}")
    return rows


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def get_empirical(row: dict[str, Any], k: int) -> float:
    empirical = row.get("empirical_pass_at_k", {})
    if isinstance(empirical, dict):
        if str(k) in empirical:
            return as_float(empirical[str(k)])
        if k in empirical:
            return as_float(empirical[k])
    return 0.0


def pairwise_same_pass1_pairs(rows: list[dict[str, Any]], tolerance: float = 0.02) -> list[dict[str, Any]]:
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_benchmark[str(row.get("benchmark_label", ""))].append(row)

    pairs: list[dict[str, Any]] = []
    for benchmark_label, rows_in_benchmark in sorted(by_benchmark.items()):
        ordered = sorted(
            rows_in_benchmark,
            key=lambda row: (as_float(row.get("alpha", 0.0)), str(row.get("model_key", ""))),
        )
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                left_alpha = as_float(left.get("alpha", 0.0))
                right_alpha = as_float(right.get("alpha", 0.0))
                alpha_gap = abs(left_alpha - right_alpha)
                if alpha_gap > tolerance:
                    continue
                left_pass4 = get_empirical(left, 4)
                right_pass4 = get_empirical(right, 4)
                pairs.append(
                    {
                        "benchmark_label": benchmark_label,
                        "left_model": str(left.get("model_key", "")),
                        "right_model": str(right.get("model_key", "")),
                        "left_display_name": str(left.get("model_display_name", "")),
                        "right_display_name": str(right.get("model_display_name", "")),
                        "alpha_gap": alpha_gap,
                        "beta_gap": abs(as_float(left.get("beta", 0.0)) - as_float(right.get("beta", 0.0))),
                        "gamma_gap": abs(as_float(left.get("gamma", 0.0)) - as_float(right.get("gamma", 0.0))),
                        "pass1_gap": alpha_gap,
                        "pass4_gap": abs(left_pass4 - right_pass4),
                        "left_alpha": left_alpha,
                        "right_alpha": right_alpha,
                        "left_beta": as_float(left.get("beta", 0.0)),
                        "right_beta": as_float(right.get("beta", 0.0)),
                        "left_gamma": as_float(left.get("gamma", 0.0)),
                        "right_gamma": as_float(right.get("gamma", 0.0)),
                        "left_pass4": left_pass4,
                        "right_pass4": right_pass4,
                    }
                )

    return sorted(
        pairs,
        key=lambda row: (
            row["benchmark_label"],
            -row["pass4_gap"],
            -row["gamma_gap"],
            row["alpha_gap"],
            row["left_model"],
            row["right_model"],
        ),
    )


def best_pair(pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not pairs:
        return None
    return max(
        pairs,
        key=lambda row: (
            row["pass4_gap"],
            row["gamma_gap"],
            -row["alpha_gap"],
            row["benchmark_label"],
            row["left_model"],
            row["right_model"],
        ),
    )


def best_pairs_by_benchmark(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[row["benchmark_label"]].append(row)
    rows: list[dict[str, Any]] = []
    for benchmark_label in sorted(grouped):
        row = best_pair(grouped[benchmark_label])
        if row is None:
            continue
        rows.append({"benchmark_label": benchmark_label, **row})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PAIR_COLUMNS})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def scope_summary(
    scope: str,
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    tolerance: float,
    *,
    pass4_gap_distribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts = Counter(row["benchmark_label"] for row in pairs)
    top = best_pair(pairs)
    by_benchmark = best_pairs_by_benchmark(pairs)
    summary: dict[str, Any] = {
        "scope": scope,
        "source_file": str(source_path(scope).resolve()),
        "psf_rows": len(rows),
        "pair_count": len(pairs),
        "benchmark_pair_counts": dict(sorted(counts.items())),
        "benchmark_count": len(counts),
        "tolerance": tolerance,
        "pairs_ge_0_10": sum(1 for row in pairs if row["pass4_gap"] >= 0.10),
        "pairs_ge_0_15": sum(1 for row in pairs if row["pass4_gap"] >= 0.15),
        "best_pair": top,
        "best_pairs_by_benchmark": by_benchmark,
    }
    if pass4_gap_distribution is not None:
        summary["pass4_gap_distribution"] = pass4_gap_distribution
    return summary


def run_scope(scope: str, tolerance: float) -> dict[str, Any]:
    from distribution import write_distribution_artifacts

    rows = load_rows(source_path(scope))
    pairs = pairwise_same_pass1_pairs(rows, tolerance=tolerance)
    out_dir = output_dir()
    write_json(out_dir / f"{scope}_pairs.json", pairs)
    write_csv(out_dir / f"{scope}_pairs.csv", pairs)
    dist_artifacts = write_distribution_artifacts(out_dir, scope, pairs)
    summary = scope_summary(
        scope,
        rows,
        pairs,
        tolerance,
        pass4_gap_distribution=dist_artifacts["distribution"],
    )
    summary["distribution_artifacts"] = {
        key: dist_artifacts[key]
        for key in ("distribution_json", "cdf_csv", "cdf_png")
        if dist_artifacts.get(key)
    }
    write_json(out_dir / f"{scope}_summary.json", summary)
    write_csv(out_dir / f"{scope}_best_pairs.csv", summary["best_pairs_by_benchmark"])
    write_json(out_dir / f"{scope}_best_pairs.json", summary["best_pairs_by_benchmark"])
    return summary


def combined_summary(group_summary: dict[str, Any] | None, variant_summary: dict[str, Any] | None) -> dict[str, Any]:
    candidates = []
    for scope_summary_payload in (group_summary, variant_summary):
        if not scope_summary_payload:
            continue
        best = scope_summary_payload.get("best_pair")
        if best:
            candidates.append({"scope": scope_summary_payload["scope"], **best})
    overall = best_pair(candidates)
    return {
        "scopes": {
            summary["scope"]: {
                "source_file": summary["source_file"],
                "psf_rows": summary["psf_rows"],
                "pair_count": summary["pair_count"],
                "benchmark_pair_counts": summary["benchmark_pair_counts"],
                "benchmark_count": summary["benchmark_count"],
                "pairs_ge_0_10": summary["pairs_ge_0_10"],
                "pairs_ge_0_15": summary["pairs_ge_0_15"],
                "best_pair": summary["best_pair"],
                "pass4_gap_distribution": summary.get("pass4_gap_distribution"),
                "paper_ready_sentence": (
                    summary.get("pass4_gap_distribution", {}).get("paper_ready_sentence")
                ),
            }
            for summary in (group_summary, variant_summary)
            if summary
        },
        "canonical_paper_ready_sentence": (
            (group_summary or {}).get("pass4_gap_distribution", {}).get("paper_ready_sentence")
        ),
        "overall_best_pair": overall,
        "notes": [
            "P2 ranks same-Pass@1 pairs by Pass@4 gap.",
            "The current cleaned-data run does not reach the 15-point headline gap from the brief; the best observed gap is about 10.46 points.",
            "Canonical paper scope: group only (63 pairs). See ANALYSIS_SPEC.md and p2_outputs/canonical_lock.json.",
        ],
        "canonical_lock": str((output_dir() / "canonical_lock.json").resolve()),
        "analysis_spec": str((project_root() / "ANALYSIS_SPEC.md").resolve()),
    }


def refresh_canonical_lock() -> None:
    from lock_canonical import build_lock_payload

    payload = build_lock_payload()
    lock_path = output_dir() / "canonical_lock.json"
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload.get("status") != "locked":
        raise SystemExit(f"canonical lock failed: see {lock_path}")
    print(f"Canonical lock: {lock_path} ({payload['canonical_counts']['pair_count']} pairs)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Squad C P2 pair analysis.")
    parser.add_argument(
        "--scope",
        choices=["group", "variant", "both"],
        default="both",
        help="Which PSF table to analyze.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Maximum absolute alpha gap for a same-Pass@1 pair.",
    )
    args = parser.parse_args()

    summaries: dict[str, Any] = {}
    if args.scope in {"group", "both"}:
        summaries["group"] = run_scope("group", args.tolerance)
    if args.scope in {"variant", "both"}:
        summaries["variant"] = run_scope("variant", args.tolerance)

    combined = combined_summary(summaries.get("group"), summaries.get("variant"))
    out_dir = output_dir()
    write_json(out_dir / "p2_summary.json", combined)

    if args.scope in {"group", "both"}:
        refresh_canonical_lock()
        from sanity_checks_p2 import run_sanity_checks

        sanity = run_sanity_checks()
        write_json(out_dir / "phase4_sanity_report.json", sanity)
        if sanity.get("status") != "passed":
            raise SystemExit("Phase 4 sanity checks failed; see phase4_sanity_report.json")

    overall = combined.get("overall_best_pair")
    print(f"P2 output directory: {out_dir}")
    if overall:
        print(
            "Best pair: "
            f"{overall['scope']} / {overall['benchmark_label']} / "
            f"{overall['left_model']} vs {overall['right_model']} "
            f"(alpha_gap={overall['alpha_gap']:.6f}, pass4_gap={overall['pass4_gap']:.6f})"
        )
    else:
        print("No qualifying same-Pass@1 pairs were found.")

    group_dist = (summaries.get("group") or {}).get("pass4_gap_distribution")
    if group_dist:
        print(f"Canonical distribution: {group_dist.get('paper_ready_sentence', '')}")


if __name__ == "__main__":
    main()
