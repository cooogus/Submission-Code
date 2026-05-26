#!/usr/bin/env python3
"""
Experiments R1-R5.

Reads: derived/metrics/rpc_sem/rpc_task2_merged.csv
       SQUAD A Results/model_identity_map.json, model_family_lookup.json
       Cleaned JSON (for R5 per-attempt costs): SQUAD A Results/cleaned_data/… or cleaned_data/…

Writes under derived/experiments/r1 … r5 and a run manifest.
Run with repo venv:  .venv/bin/python scripts/run_task3_experiments.py
"""

from __future__ import annotations

import ast
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
MERGED_CSV = REPO / "derived" / "metrics" / "rpc_sem" / "rpc_task2_merged.csv"
IDENTITY_JSON = REPO / "SQUAD A Results" / "model_identity_map.json"
FAMILY_JSON = REPO / "SQUAD A Results" / "model_family_lookup.json"

CLEANED_ROOTS = [
    REPO / "SQUAD A Results" / "cleaned_data",
    REPO / "cleaned_data",
]

# Brief / squad messaging: reasoning-capable slugs (lowercase match)
REASONING_SLUGS = frozenset(
    {
        "o4-mini",
        "grok-4-fast-reasoning",
        "deepseek_r1",
    }
)

R1_OUT = REPO / "derived" / "experiments" / "r1"
R2_OUT = REPO / "derived" / "experiments" / "r2"
R3_OUT = REPO / "derived" / "experiments" / "r3"
R4_OUT = REPO / "derived" / "experiments" / "r4"
R5_OUT = REPO / "derived" / "experiments" / "r5"


def resolve_cleaned_json(source_path: str) -> Path | None:
    """source_path like cleaned_data/agieval/lsat_ar/model/1.json"""
    if "cleaned_data/" not in source_path:
        return None
    tail = source_path.split("cleaned_data/", 1)[1]
    for base in CLEANED_ROOTS:
        p = base / tail
        if p.is_file():
            return p
    return None


def first_attempt_correct(correct_str: str) -> float | None:
    try:
        arr = ast.literal_eval(correct_str)
        if not arr:
            return None
        return 1.0 if bool(arr[0]) else 0.0
    except (ValueError, SyntaxError):
        return None


def per_attempt_costs(data: dict) -> list[float] | None:
    if "detailed_costs" in data and isinstance(data["detailed_costs"], list):
        out = []
        for item in data["detailed_costs"]:
            if item is None:
                return None
            if isinstance(item, dict) and "cost" in item:
                out.append(float(item["cost"]))
            else:
                return None
        if len(out) == 4:
            return out
    return None


def load_strict_frame() -> pd.DataFrame:
    df = pd.read_csv(MERGED_CSV)
    m = (
        (df["has_rpc_s"].astype(str) == "1")
        & (df["has_rpc_l"].astype(str) == "1")
        & df["rpc_sem"].notna()
        & (df["rpc_sem"].astype(str).str.strip() != "")
    )
    df = df.loc[m].copy()
    df["rpc_sem_f"] = pd.to_numeric(df["rpc_sem"], errors="coerce")
    df["rpc_s_f"] = pd.to_numeric(df["rpc_s"], errors="coerce")
    df["rpc_l_f"] = pd.to_numeric(df["rpc_l_cot_ngram_mean_intersection"], errors="coerce")
    df["first_correct"] = df["correct"].map(first_attempt_correct)
    return df


def corr_pearson_spearman(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"n": int(len(x)), "pearson_r": math.nan, "pearson_p": math.nan}
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {
        "n": int(len(x)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }


def experiment_r1(df: pd.DataFrame) -> None:
    R1_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    pairs = [
        ("rpc_l_f", "rpc_sem_f", "rpc_l_vs_rpc_sem"),
        ("rpc_l_f", "rpc_s_f", "rpc_l_vs_rpc_s"),
        ("rpc_s_f", "rpc_sem_f", "rpc_s_vs_rpc_sem"),
    ]
    for a, b, name in pairs:
        d = corr_pearson_spearman(df[a].to_numpy(), df[b].to_numpy())
        rows.append({"pair": name, "x": a, "y": b, **d})

    pd.DataFrame(rows).to_csv(R1_OUT / "metric_metric_correlations.csv", index=False)

    human_path = REPO / "derived" / "experiments" / "r1" / "human_labels.csv"
    human_note = {
        "status": "skipped",
        "reason": "No human_labels.csv found. Add derived/experiments/r1/human_labels.csv "
        "with columns: source_path, human_same_approach_mean (0–1 or majority vote) "
        "to compute human–metric correlations.",
        "expected_path": str(human_path.relative_to(REPO)),
    }
    if human_path.is_file():
        h = pd.read_csv(human_path)
        if "source_path" in h.columns and "human_same_approach_mean" in h.columns:
            m = df.merge(h, on="source_path", how="inner")
            if len(m) >= 5:
                hu = m["human_same_approach_mean"].to_numpy(dtype=float)
                human_rows = []
                for col, label in [
                    ("rpc_l_f", "human_vs_rpc_l"),
                    ("rpc_s_f", "human_vs_rpc_s"),
                    ("rpc_sem_f", "human_vs_rpc_sem"),
                ]:
                    human_rows.append(
                        {"pair": label, **corr_pearson_spearman(hu, m[col].to_numpy())}
                    )
                pd.DataFrame(human_rows).to_csv(
                    R1_OUT / "human_metric_correlations.csv", index=False
                )
                human_note = {
                    "status": "ok",
                    "n_matched_rows": int(len(m)),
                }

    with open(R1_OUT / "r1_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric_correlations_csv": "derived/experiments/r1/metric_metric_correlations.csv",
                "human_validation": human_note,
            },
            f,
            indent=2,
        )


def experiment_r2(df: pd.DataFrame, pass1_tol: float = 0.02) -> None:
    R2_OUT.mkdir(parents=True, exist_ok=True)
    # Pass@1 per (benchmark, model)
    g = df.groupby(["benchmark", "model"], as_index=False).agg(
        pass_at_1=("first_correct", "mean"),
        n=("first_correct", "count"),
    )
    benchmarks = g["benchmark"].unique()
    pair_rows = []
    for bench in benchmarks:
        sub = g[g["benchmark"] == bench].copy()
        sub = sub[np.isfinite(sub["pass_at_1"]) & (sub["n"] > 0)]
        models = sub["model"].tolist()
        passv = sub.set_index("model")["pass_at_1"].to_dict()
        n_q = sub.set_index("model")["n"].to_dict()
        for i, m1 in enumerate(models):
            for m2 in models[i + 1 :]:
                p1, p2 = passv[m1], passv[m2]
                if not (np.isfinite(p1) and np.isfinite(p2)):
                    continue
                if abs(p1 - p2) > pass1_tol:
                    continue
                if n_q[m1] < 1 or n_q[m2] < 1:
                    continue
                d1 = df[(df["benchmark"] == bench) & (df["model"] == m1)]
                d2 = df[(df["benchmark"] == bench) & (df["model"] == m2)]
                x_sem, y_sem = d1["rpc_sem_f"].to_numpy(), d2["rpc_sem_f"].to_numpy()
                x_s, y_s = d1["rpc_s_f"].to_numpy(), d2["rpc_s_f"].to_numpy()
                x_l, y_l = d1["rpc_l_f"].to_numpy(), d2["rpc_l_f"].to_numpy()
                # Mann–Whitney on semantic (independent samples, different questions)
                mw_sem = stats.mannwhitneyu(
                    x_sem, y_sem, alternative="two-sided"
                ) if len(x_sem) > 2 and len(y_sem) > 2 else None
                mw_s = (
                    stats.mannwhitneyu(x_s, y_s, alternative="two-sided")
                    if len(x_s) > 2 and len(y_s) > 2
                    else None
                )
                mw_l = (
                    stats.mannwhitneyu(x_l, y_l, alternative="two-sided")
                    if len(x_l) > 2 and len(y_l) > 2
                    else None
                )
                pair_rows.append(
                    {
                        "benchmark": bench,
                        "model_a": m1,
                        "model_b": m2,
                        "pass_at_1_a": p1,
                        "pass_at_1_b": p2,
                        "pass_at_1_diff": abs(p1 - p2),
                        "n_questions_a": int(n_q[m1]),
                        "n_questions_b": int(n_q[m2]),
                        "mean_rpc_sem_a": float(np.nanmean(x_sem)),
                        "mean_rpc_sem_b": float(np.nanmean(y_sem)),
                        "mean_rpc_sem_diff": float(
                            np.nanmean(x_sem) - np.nanmean(y_sem)
                        ),
                        "mw_sem_statistic": float(mw_sem.statistic) if mw_sem else math.nan,
                        "mw_sem_pvalue": float(mw_sem.pvalue) if mw_sem else math.nan,
                        "mean_rpc_s_a": float(np.nanmean(x_s)),
                        "mean_rpc_s_b": float(np.nanmean(y_s)),
                        "mw_s_pvalue": float(mw_s.pvalue) if mw_s else math.nan,
                        "mean_rpc_l_a": float(np.nanmean(x_l)),
                        "mean_rpc_l_b": float(np.nanmean(y_l)),
                        "mw_l_pvalue": float(mw_l.pvalue) if mw_l else math.nan,
                    }
                )

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(R2_OUT / "pass1_matched_model_pairs.csv", index=False)
    # Headline: largest semantic mean gap among near-tied pass@1 pairs
    if len(pair_df):
        pair_df["abs_sem_diff"] = pair_df["mean_rpc_sem_diff"].abs()
        pair_df.sort_values("abs_sem_diff", ascending=False).head(25).to_csv(
            R2_OUT / "top_semantic_divergence_pairs.csv", index=False
        )


def slug_family_maps() -> tuple[dict[str, str], dict[str, str]]:
    with open(IDENTITY_JSON, encoding="utf-8") as f:
        slug_to_display = json.load(f)
    with open(FAMILY_JSON, encoding="utf-8") as f:
        display_to_family = json.load(f)
    slug_to_family: dict[str, str] = {}
    for slug, disp in slug_to_display.items():
        slug_to_family[slug.lower()] = display_to_family.get(disp, "Unknown")
    return slug_to_display, slug_to_family


def experiment_r3(df: pd.DataFrame) -> None:
    R3_OUT.mkdir(parents=True, exist_ok=True)
    _, slug_to_family = slug_family_maps()
    df = df.copy()
    df["family"] = df["model"].str.lower().map(lambda s: slug_to_family.get(s, "Unknown"))
    df["reasoning_mode"] = df["model"].str.lower().map(
        lambda s: "reasoning" if s in REASONING_SLUGS else "non_reasoning"
    )

    fam_bench = (
        df.groupby(["family", "benchmark"], as_index=False)
        .agg(
            mean_rpc_sem=("rpc_sem_f", "mean"),
            mean_rpc_s=("rpc_s_f", "mean"),
            mean_rpc_l=("rpc_l_f", "mean"),
            n=("rpc_sem_f", "count"),
        )
    )
    fam_bench.to_csv(R3_OUT / "rpc_by_family_benchmark.csv", index=False)

    rm = (
        df.groupby("reasoning_mode", as_index=False)
        .agg(
            mean_rpc_sem=("rpc_sem_f", "mean"),
            mean_rpc_s=("rpc_s_f", "mean"),
            mean_rpc_l=("rpc_l_f", "mean"),
            n=("rpc_sem_f", "count"),
        )
    )
    rm.to_csv(R3_OUT / "rpc_by_reasoning_mode.csv", index=False)


def experiment_r4(df: pd.DataFrame) -> None:
    R4_OUT.mkdir(parents=True, exist_ok=True)
    pivot_sem = df.pivot_table(
        index="model",
        columns="benchmark",
        values="rpc_sem_f",
        aggfunc="mean",
    )
    pivot_sem.to_csv(R4_OUT / "heatmap_rpc_sem_model_x_benchmark.csv")
    df.pivot_table(
        index="model", columns="benchmark", values="rpc_s_f", aggfunc="mean"
    ).to_csv(R4_OUT / "heatmap_rpc_s_model_x_benchmark.csv")
    df.pivot_table(
        index="model", columns="benchmark", values="rpc_l_f", aggfunc="mean"
    ).to_csv(R4_OUT / "heatmap_rpc_l_model_x_benchmark.csv")


def experiment_r5(df: pd.DataFrame, n_boot: int = 500, seed: int = 42) -> None:
    R5_OUT.mkdir(parents=True, exist_ok=True)
    variances = []
    means = []
    paths = []
    for _, row in df.iterrows():
        sp = row["source_path"]
        p = resolve_cleaned_json(sp)
        if p is None:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        costs = per_attempt_costs(data)
        if costs is None:
            continue
        variances.append(float(np.var(costs, ddof=1)))
        means.append(float(np.mean(costs)))
        paths.append(sp)

    sub = df[df["source_path"].isin(paths)].copy()
    sub["cost_var_across_attempts"] = sub["source_path"].map(
        dict(zip(paths, variances))
    )
    sub["cost_mean_across_attempts"] = sub["source_path"].map(
        dict(zip(paths, means))
    )
    sub.to_csv(R5_OUT / "strict_rows_with_cost_variance.csv", index=False)

    cv = sub["cost_var_across_attempts"].to_numpy(dtype=float)
    rows = []
    for col, name in [
        ("rpc_sem_f", "cost_var_vs_rpc_sem"),
        ("rpc_s_f", "cost_var_vs_rpc_s"),
        ("rpc_l_f", "cost_var_vs_rpc_l"),
    ]:
        y = sub[col].to_numpy(dtype=float)
        rows.append({"pair": name, **corr_pearson_spearman(cv, y)})
    pd.DataFrame(rows).to_csv(R5_OUT / "correlation_cost_variance_rpc.csv", index=False)

    rng = np.random.default_rng(seed)
    boot = []
    x, y = cv, sub["rpc_sem_f"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) > 10:
        for _ in range(n_boot):
            idx = rng.integers(0, len(x), size=len(x))
            r, _ = stats.pearsonr(x[idx], y[idx])
            boot.append(float(r))
        lo, hi = np.quantile(boot, [0.025, 0.975])
        boot_summary = {
            "pearson_rpc_sem_cost_var_n": int(len(x)),
            "bootstrap_mean_r": float(np.mean(boot)),
            "bootstrap_ci95_low": float(lo),
            "bootstrap_ci95_high": float(hi),
            "n_bootstrap": n_boot,
        }
    else:
        boot_summary = {"note": "too few rows with detailed_costs"}

    with open(R5_OUT / "r5_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_rows_with_cost_variance": int(len(sub)),
                "n_strict_input_rows": int(len(df)),
                "bootstrap_semantic": boot_summary,
            },
            f,
            indent=2,
        )


def main() -> int:
    if not MERGED_CSV.is_file():
        print(f"Missing {MERGED_CSV}", file=sys.stderr)
        return 1
    df = load_strict_frame()
    if len(df) < 10:
        print("Too few strict rows for experiments.", file=sys.stderr)
        return 1

    experiment_r1(df)
    experiment_r2(df)
    experiment_r3(df)
    experiment_r4(df)
    experiment_r5(df)

    manifest = {
        "strict_row_count": int(len(df)),
        "merged_csv": str(MERGED_CSV.relative_to(REPO)),
        "outputs": {
            "r1": "derived/experiments/r1/",
            "r2": "derived/experiments/r2/",
            "r3": "derived/experiments/r3/",
            "r4": "derived/experiments/r4/",
            "r5": "derived/experiments/r5/",
        },
        "interpreter": "Use .venv/bin/python scripts/run_task3_experiments.py",
    }
    out_root = REPO / "derived" / "experiments"
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "task3_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Task 3 experiments done. Strict n={len(df)}. See derived/experiments/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
