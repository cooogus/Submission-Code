#!/usr/bin/env python3
"""
Merge RPC-Sem with Squad A RPC-S + RPC-L (from n_grams.jsonl).

Reads:
  - derived/metrics/rpc_sem/rpc_sem.csv
  - SQUAD A Results/rpc_s_structural_scores.csv
  - SQUAD A Results/rpc-l/n_grams.jsonl

Writes:
  - derived/metrics/rpc_sem/rpc_task2_merged.csv
  - derived/metrics/rpc_sem/task2_join_manifest.json

RPC-L: mean of 6 pairwise len_overlap values (3-gram intersection counts on CoT
text) per json_path — same construction as Squad A's n_gram export, not Jaccard.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RPC_SEM_CSV = REPO_ROOT / "derived" / "metrics" / "rpc_sem" / "rpc_sem.csv"
RPC_S_CSV = REPO_ROOT / "SQUAD A Results" / "rpc_s_structural_scores.csv"
NGRAMS_PATH = REPO_ROOT / "SQUAD A Results" / "rpc-l" / "n_grams.jsonl"
OUT_MERGED = REPO_ROOT / "derived" / "metrics" / "rpc_sem" / "rpc_task2_merged.csv"
OUT_MANIFEST = REPO_ROOT / "derived" / "metrics" / "rpc_sem" / "task2_join_manifest.json"

# rpc_s columns to copy (exclude long seq_* strings)
RPC_S_COPY = [
    "benchmark",
    "model",
    "question_id",
    "rpc_s",
    "pair_1_2",
    "pair_1_3",
    "pair_1_4",
    "pair_2_3",
    "pair_2_4",
    "pair_3_4",
    "correct",
    "answers",
    "gold_answer",
]


def json_path_from_ngram_record(r: dict) -> str:
    b, bt, m, idx = r["benchmark"], r["benchmark_type"], r["model"], r["idx"]
    if b == "agieval":
        return f"cleaned_data/agieval/{bt}/{m}/{idx}.json"
    return f"cleaned_data/{b}/{bt}/{m}/{idx}.json"


def load_rpc_l_means() -> dict[str, float]:
    """Mean len_overlap over 6 cot pairs per json_path (only paths with 6 lines)."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    with open(NGRAMS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            jp = json_path_from_ngram_record(r)
            sums[jp] += float(r["len_overlap"])
            counts[jp] += 1

    out: dict[str, float] = {}
    for jp, c in counts.items():
        if c == 6:
            out[jp] = sums[jp] / 6.0
    return out


def load_rpc_s_by_path() -> dict[str, dict[str, str]]:
    by_path: dict[str, dict[str, str]] = {}
    with open(RPC_S_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_path[row["json_path"]] = {k: row.get(k, "") for k in RPC_S_COPY}
    return by_path


def main() -> int:
    if not RPC_SEM_CSV.is_file():
        print(f"ERROR: missing {RPC_SEM_CSV}", file=sys.stderr)
        return 1
    if not RPC_S_CSV.is_file():
        print(f"ERROR: missing {RPC_S_CSV}", file=sys.stderr)
        return 1
    if not NGRAMS_PATH.is_file():
        print(f"ERROR: missing {NGRAMS_PATH}", file=sys.stderr)
        return 1

    rpc_l = load_rpc_l_means()
    rpc_s_by_path = load_rpc_s_by_path()

    sem_fieldnames = []
    with open(RPC_SEM_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sem_fieldnames = reader.fieldnames or []
        sem_rows = list(reader)

    extra = [
        "has_rpc_s",
        "has_rpc_l",
        "rpc_l_cot_ngram_mean_intersection",
    ] + RPC_S_COPY

    out_fields = [f for f in sem_fieldnames if f is not None] + extra

    n_sem_scored = 0
    n_join_s = 0
    n_join_l = 0
    n_join_both = 0

    out_rows: list[dict[str, str]] = []
    for row in sem_rows:
        jp = row.get("source_path", "")
        has_s = jp in rpc_s_by_path
        has_l = jp in rpc_l
        if row.get("rpc_sem", "").strip():
            n_sem_scored += 1
            if has_s:
                n_join_s += 1
            if has_l:
                n_join_l += 1
            if has_s and has_l:
                n_join_both += 1

        ext: dict[str, str] = {
            "has_rpc_s": "1" if has_s else "0",
            "has_rpc_l": "1" if has_l else "0",
            "rpc_l_cot_ngram_mean_intersection": (
                f"{rpc_l[jp]:.6f}" if has_l else ""
            ),
        }
        if has_s:
            ext.update(rpc_s_by_path[jp])
        else:
            for k in RPC_S_COPY:
                ext[k] = ""

        merged = {**row, **ext}
        out_rows.append({k: merged.get(k, "") for k in out_fields})

    OUT_MERGED.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MERGED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/finish_task2_merge_metrics.py",
        "inputs": {
            "rpc_sem_csv": str(RPC_SEM_CSV.relative_to(REPO_ROOT)),
            "rpc_s_csv": str(RPC_S_CSV.relative_to(REPO_ROOT)),
            "n_grams_jsonl": str(NGRAMS_PATH.relative_to(REPO_ROOT)),
        },
        "output_merged_csv": str(OUT_MERGED.relative_to(REPO_ROOT)),
        "rpc_l_note": "rpc_l_cot_ngram_mean_intersection = mean of 6 pairwise "
        "len_overlap values from n_grams.jsonl (CoT 3-gram intersection counts); "
        "not Jaccard—document clearly in methods if used.",
        "counts": {
            "rpc_sem_rows_total": len(sem_rows),
            "rpc_sem_rows_with_score": n_sem_scored,
            "unique_json_paths_rpc_l_from_ngrams": len(rpc_l),
            "rpc_s_rows": len(rpc_s_by_path),
            "among_scored_rpc_sem_has_rpc_s": n_join_s,
            "among_scored_rpc_sem_has_rpc_l": n_join_l,
            "among_scored_rpc_sem_has_both": n_join_both,
        },
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {OUT_MERGED}")
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
