from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .io import safe_mean, safe_variance
from .registry import infer_family
from .stats import pearson_correlation


def _euclidean_sq(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _standardize(points: Sequence[Sequence[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    if not points:
        return [], [], []
    dims = len(points[0])
    means = [sum(point[d] for point in points) / len(points) for d in range(dims)]
    stds: List[float] = []
    for d in range(dims):
        values = [point[d] for point in points]
        mean = means[d]
        var = sum((value - mean) ** 2 for value in values) / len(values)
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    scaled = [[(point[d] - means[d]) / stds[d] for d in range(dims)] for point in points]
    return scaled, means, stds


def _choose_kmeanspp(points: Sequence[Sequence[float]], k: int, rng: random.Random) -> list[list[float]]:
    centroids = [list(points[rng.randrange(len(points))])]
    while len(centroids) < k:
        distances = [min(_euclidean_sq(point, centroid) for centroid in centroids) for point in points]
        total = sum(distances)
        if total <= 1e-12:
            centroids.append(list(points[rng.randrange(len(points))]))
            continue
        cutoff = rng.random() * total
        running = 0.0
        for point, distance in zip(points, distances):
            running += distance
            if running >= cutoff:
                centroids.append(list(point))
                break
        else:
            centroids.append(list(points[-1]))
    return centroids


def kmeans(points: Sequence[Sequence[float]], k: int = 3, restarts: int = 20, seed: int = 0, max_iter: int = 100) -> dict[str, Any]:
    if not points:
        return {"status": "empty", "labels": [], "centroids": [], "inertia": 0.0, "scaled_points": []}
    if len(points) <= k:
        labels = list(range(len(points)))
        return {
            "status": "degenerate",
            "labels": labels,
            "centroids": [list(point) for point in points],
            "inertia": 0.0,
            "scaled_points": [list(point) for point in points],
        }

    scaled_points, means, stds = _standardize(points)
    best = None
    rng = random.Random(seed)

    for restart in range(max(1, restarts)):
        local_rng = random.Random(rng.randint(0, 10**9))
        centroids = _choose_kmeanspp(scaled_points, k, local_rng)
        labels = [0] * len(scaled_points)
        for _ in range(max_iter):
            changed = False
            for idx, point in enumerate(scaled_points):
                best_label = min(range(k), key=lambda label: _euclidean_sq(point, centroids[label]))
                if labels[idx] != best_label:
                    labels[idx] = best_label
                    changed = True
            new_centroids: list[list[float]] = []
            for label in range(k):
                cluster_points = [point for point, assigned in zip(scaled_points, labels) if assigned == label]
                if not cluster_points:
                    new_centroids.append(list(scaled_points[local_rng.randrange(len(scaled_points))]))
                    continue
                dims = len(cluster_points[0])
                new_centroids.append([sum(point[d] for point in cluster_points) / len(cluster_points) for d in range(dims)])
            centroids = new_centroids
            if not changed:
                break
        inertia = sum(_euclidean_sq(point, centroids[label]) for point, label in zip(scaled_points, labels))
        candidate = {
            "labels": labels,
            "centroids": centroids,
            "inertia": inertia,
            "scaled_points": scaled_points,
            "means": means,
            "stds": stds,
        }
        if best is None or inertia < best["inertia"]:
            best = candidate

    assert best is not None
    best["status"] = "ok"
    return best


def cluster_psf_rows(rows: Sequence[dict[str, Any]], k: int = 3, seed: int = 0) -> dict[str, Any]:
    points = [[float(row.get("alpha", 0.0)), float(row.get("beta", 0.0)), float(row.get("gamma", 0.0))] for row in rows]
    clustering = kmeans(points, k=k, seed=seed)
    labels = clustering.get("labels", [])
    enriched = []
    for row, label, point in zip(rows, labels, points):
        item = dict(row)
        item["cluster"] = int(label)
        item["point"] = point
        item["family"] = infer_family(str(row.get("model_key", "")), str(row.get("model_display_name", "")))
        enriched.append(item)

    cluster_counts: Dict[int, int] = Counter(labels)
    family_counts: Dict[int, Counter] = defaultdict(Counter)
    benchmark_counts: Dict[int, Counter] = defaultdict(Counter)
    for item in enriched:
        cluster = int(item["cluster"])
        family_counts[cluster][item["family"]] += 1
        benchmark_counts[cluster][str(item.get("benchmark_label") or item.get("benchmark_group") or "")] += 1

    cluster_summary = []
    for cluster in sorted(cluster_counts):
        fam_counter = family_counts[cluster]
        bench_counter = benchmark_counts[cluster]
        cluster_summary.append(
            {
                "cluster": cluster,
                "count": cluster_counts[cluster],
                "dominant_family": fam_counter.most_common(1)[0][0] if fam_counter else "unknown",
                "family_counts": dict(fam_counter),
                "benchmark_counts": dict(bench_counter),
                "centroid": clustering["centroids"][cluster],
            }
        )
    return {
        "status": clustering.get("status", "ok"),
        "k": k,
        "labels": labels,
        "inertia": clustering.get("inertia", 0.0),
        "cluster_summary": cluster_summary,
        "rows": enriched,
    }


def cluster_psf_rows_by_benchmark(rows: Sequence[dict[str, Any]], k: int = 3, seed: int = 0) -> dict[str, Any]:
    benchmark_rows: Dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        benchmark = str(row.get("benchmark_label") or row.get("benchmark_group") or "unknown")
        benchmark_rows[benchmark].append(row)

    benchmark_clusters: dict[str, dict[str, Any]] = {}
    benchmark_summaries = []
    all_rows: list[dict[str, Any]] = []

    for benchmark in sorted(benchmark_rows):
        clustered = cluster_psf_rows(benchmark_rows[benchmark], k=k, seed=seed)
        benchmark_clusters[benchmark] = clustered
        benchmark_summaries.append(
            {
                "benchmark": benchmark,
                "count": len(benchmark_rows[benchmark]),
                "status": clustered.get("status", "ok"),
                "cluster_summary": clustered.get("cluster_summary", []),
            }
        )
        all_rows.extend(clustered.get("rows", []))

    return {
        "status": "ok" if benchmark_clusters else "empty",
        "k": k,
        "benchmark_summaries": benchmark_summaries,
        "by_benchmark": benchmark_clusters,
        "rows": all_rows,
    }
