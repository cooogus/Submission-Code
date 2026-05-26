#!/usr/bin/env python3
"""
Compute RPC-Sem from cached embeddings (Task 2).

Reads embeddings/**/*.npy (+ optional sibling *.meta.json), writes
derived/metrics/rpc_sem/rpc_sem.csv and a small run_manifest.json.

Assumes L2-normalized rows so cosine similarity equals dot product.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_ROOT = REPO_ROOT / "embeddings"
OUT_DIR = REPO_ROOT / "derived" / "metrics" / "rpc_sem"
EXPECTED_DIM = 384
EXPECTED_ATTEMPTS = 6  # pairs among 4 attempts


def benchmark_path_to_slug(benchmark_path: str) -> str:
    """
    Match Squad A rpc_s_structural_scores.csv `benchmark` column.

    - agieval/lsat_ar -> agieval_lsat_ar
    - aime/aime_2025 -> aime_2025
    - gpqa/gpqa_diamond -> gpqa_diamond
    - mmmu/mmmu_accounting -> mmmu_accounting
    """
    parts = benchmark_path.split("/")
    if len(parts) == 2 and parts[1].startswith(parts[0] + "_"):
        return parts[1]
    if len(parts) == 2:
        return f"{parts[0]}_{parts[1]}"
    return "_".join(parts)


def pairwise_cosines_normalized(V: np.ndarray) -> tuple[list[float], float]:
    """V: (4, d) L2-normalized rows -> 6 similarities and their mean."""
    sims: list[float] = []
    for i in range(4):
        for j in range(i + 1, 4):
            sims.append(float(np.dot(V[i], V[j])))
    return sims, sum(sims) / len(sims)


def parse_embedding_path(npy_path: Path) -> tuple[str, str, str]:
    """
    Infer benchmark_path, model_id, question_idx from
    embeddings/<benchmark...>/<model>/<idx>.npy
    """
    rel = npy_path.relative_to(EMBEDDINGS_ROOT)
    parts = rel.parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected layout: {npy_path}")
    question_idx = npy_path.stem
    model_id = parts[-2]
    benchmark_path = "/".join(parts[:-2])
    return benchmark_path, model_id, question_idx


def load_meta(npy_path: Path) -> dict | None:
    meta_path = npy_path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if not EMBEDDINGS_ROOT.is_dir():
        print(f"ERROR: embeddings root not found: {EMBEDDINGS_ROOT}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "rpc_sem.csv"

    fieldnames = [
        "source_path",
        "embedding_path",
        "benchmark_path",
        "benchmark_slug",
        "model_id",
        "question_idx",
        "rpc_sem",
        "skip_reason",
        "embedder",
        "pair_01",
        "pair_02",
        "pair_03",
        "pair_12",
        "pair_13",
        "pair_23",
    ]

    rows: list[dict] = []
    n_ok = 0
    n_skip = 0

    npy_files = sorted(EMBEDDINGS_ROOT.rglob("*.npy"))
    for npy_path in npy_files:
        rel_emb = str(npy_path.relative_to(REPO_ROOT))
        meta = load_meta(npy_path)

        try:
            benchmark_path, model_id, question_idx = parse_embedding_path(npy_path)
        except ValueError as e:
            rows.append(
                {
                    "source_path": "",
                    "embedding_path": rel_emb,
                    "benchmark_path": "",
                    "benchmark_slug": "",
                    "model_id": "",
                    "question_idx": "",
                    "rpc_sem": "",
                    "skip_reason": f"path_layout:{e}",
                    "embedder": "",
                    "pair_01": "",
                    "pair_02": "",
                    "pair_03": "",
                    "pair_12": "",
                    "pair_13": "",
                    "pair_23": "",
                }
            )
            n_skip += 1
            continue

        source_path = meta.get("source_path", "") if meta else ""
        embedder = meta.get("embedder", "") if meta else ""
        if not source_path:
            source_path = f"cleaned_data/{benchmark_path}/{model_id}/{question_idx}.json"

        row_base = {
            "source_path": source_path,
            "embedding_path": rel_emb,
            "benchmark_path": benchmark_path,
            "benchmark_slug": benchmark_path_to_slug(benchmark_path),
            "model_id": model_id,
            "question_idx": question_idx,
            "embedder": embedder,
        }

        try:
            V = np.load(npy_path)
        except Exception as e:
            rows.append(
                {
                    **row_base,
                    "rpc_sem": "",
                    "skip_reason": f"load_error:{e}",
                    "pair_01": "",
                    "pair_02": "",
                    "pair_03": "",
                    "pair_12": "",
                    "pair_13": "",
                    "pair_23": "",
                }
            )
            n_skip += 1
            continue

        if V.ndim != 2:
            rows.append(
                {
                    **row_base,
                    "rpc_sem": "",
                    "skip_reason": f"wrong_ndim:{V.ndim}",
                    "pair_01": "",
                    "pair_02": "",
                    "pair_03": "",
                    "pair_12": "",
                    "pair_13": "",
                    "pair_23": "",
                }
            )
            n_skip += 1
            continue

        n_attempts, dim = V.shape
        if dim != EXPECTED_DIM:
            rows.append(
                {
                    **row_base,
                    "rpc_sem": "",
                    "skip_reason": f"wrong_dim:{dim}",
                    "pair_01": "",
                    "pair_02": "",
                    "pair_03": "",
                    "pair_12": "",
                    "pair_13": "",
                    "pair_23": "",
                }
            )
            n_skip += 1
            continue

        if n_attempts != 4:
            rows.append(
                {
                    **row_base,
                    "rpc_sem": "",
                    "skip_reason": f"wrong_n_attempts:{n_attempts}",
                    "pair_01": "",
                    "pair_02": "",
                    "pair_03": "",
                    "pair_12": "",
                    "pair_13": "",
                    "pair_23": "",
                }
            )
            n_skip += 1
            continue

        sims, mean_sim = pairwise_cosines_normalized(V.astype(np.float64, copy=False))
        rows.append(
            {
                **row_base,
                "rpc_sem": f"{mean_sim:.10f}",
                "skip_reason": "",
                "pair_01": f"{sims[0]:.10f}",
                "pair_02": f"{sims[1]:.10f}",
                "pair_03": f"{sims[2]:.10f}",
                "pair_12": f"{sims[3]:.10f}",
                "pair_13": f"{sims[4]:.10f}",
                "pair_23": f"{sims[5]:.10f}",
            }
        )
        n_ok += 1

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/compute_rpc_sem.py",
        "embeddings_root": str(EMBEDDINGS_ROOT.relative_to(REPO_ROOT)),
        "output_csv": str(out_csv.relative_to(REPO_ROOT)),
        "n_embedding_files": len(npy_files),
        "n_rpc_sem_computed": n_ok,
        "n_skipped": n_skip,
        "expected_shape": [4, EXPECTED_DIM],
        "pair_count": EXPECTED_ATTEMPTS,
        "note": "rpc_sem = mean of 6 pairwise cosines; vectors assumed L2-normalized (Task 1).",
    }
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {out_csv} ({n_ok} computed, {n_skip} skipped, {len(npy_files)} npy files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
