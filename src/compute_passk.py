"""
compute_passk.py
Usage: python compute_passk.py --data path/to/cleaned_data
Output: passk.csv
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

K_MAX = 4


def get_benchmark(benchmark_type):
    if benchmark_type.startswith("mmmu"):  return "mmmu"
    if benchmark_type.startswith("gpqa"):  return "gpqa"
    if benchmark_type.startswith("lsat"):  return "agieval"
    if benchmark_type.startswith("aime"):  return "aime"
    return benchmark_type


def run(data_dir):
    counts = defaultdict(lambda: defaultdict(list))
    skipped = 0

    for path in Path(data_dir).rglob("*.json"):
        try:
            data  = json.loads(path.read_text(encoding="utf-8"))
            parts = re.findall(r"[^/\\]+", str(path))
            benchmark = get_benchmark(parts[-3])
            model     = parts[-2]

            correct = data.get("correct", [])
            if not correct:
                skipped += 1
                continue

            counts[model][benchmark].append(correct[:K_MAX])

        except Exception as e:
            print(f"[skip] {path}: {e}")

    rows = []
    for model in sorted(counts):
        for benchmark in sorted(counts[model]):
            vecs = counts[model][benchmark]
            n    = len(vecs)
            passk = []
            for k in range(1, K_MAX + 1):
                hit = sum(any(v[j] for j in range(min(k, len(v)))) for v in vecs)
                passk.append(round(hit / n, 6))
            rows.append({
                "model":       model,
                "benchmark":   benchmark,
                "n_questions": n,
                "pass_at_1":   passk[0],
                "pass_at_2":   passk[1],
                "pass_at_3":   passk[2],
                "pass_at_4":   passk[3],
            })
            print(f"  {model:<30} {benchmark:<12}  n={n:<5}"
                  f"  Pass@1={passk[0]:.3f}  Pass@4={passk[3]:.3f}")

    if skipped:
        print(f"\n[warn] {skipped} files skipped (no 'correct' field)")

    df = pd.DataFrame(rows)
    df.to_csv("passk.csv", index=False)
    print(f"\nSaved {len(df)} (model, benchmark) pairs → passk.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to cleaned_data folder")
    args = parser.parse_args()
    run(args.data)
