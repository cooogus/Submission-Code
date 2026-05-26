from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .cluster import cluster_psf_rows, cluster_psf_rows_by_benchmark
from .cross_analysis import build_cross_analysis, load_rpc_rows
from .io import dump_json, ensure_dir, safe_mean
from .loaders import load_all_records
from .psf import build_psf_table, pairwise_same_pass1_pairs
from .registry import build_model_registry, default_repo_root, load_benchmark_groups
from .schema import QuestionRecord


def _project_root(repo_root: Path) -> Path:
    return repo_root / "SQUADC"


def _artifact_dir(repo_root: Path) -> Path:
    return _project_root(repo_root) / "artifacts"


def build_manifest(repo_root: Path | None = None, preserve_messages: bool = False) -> dict[str, Any]:
    repo_root = repo_root or default_repo_root()
    project_dir = _project_root(repo_root)
    artifact_dir = ensure_dir(_artifact_dir(repo_root))
    records, registry = load_all_records(repo_root=repo_root, preserve_messages=preserve_messages)
    groups = load_benchmark_groups(project_dir)

    benchmark_counts: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"records": 0, "psf_ready": 0, "variants": Counter(), "models": Counter()})
    model_counts: Counter[str] = Counter()
    issues: List[str] = []

    for record in records:
        group = record.benchmark_group
        benchmark_counts[group]["records"] += 1
        benchmark_counts[group]["variants"][record.benchmark_variant] += 1
        benchmark_counts[group]["models"][record.model_key] += 1
        model_counts[record.model_key] += 1
        if record.psf_ready:
            benchmark_counts[group]["psf_ready"] += 1
        if record.metadata.get("contains_images"):
            benchmark_counts[group].setdefault("multimodal_records", 0)
            benchmark_counts[group]["multimodal_records"] += 1
        if len(record.attempts) < 4 and record.benchmark_group != "aider":
            issues.append(f"{record.source_path} has only {len(record.attempts)} attempts")
        if record.benchmark_group == "aider" and len(record.attempts) < 4:
            issues.append(f"Aider record is single-attempt only: {record.source_path}")

    summary = {
        "repo_root": str(repo_root),
        "project_root": str(project_dir),
        "groups": [group.to_dict() for group in groups],
        "records_total": len(records),
        "psf_ready_total": sum(1 for record in records if record.psf_ready),
        "model_registry_size": len(registry),
        "model_registry": {key: entry.to_dict() for key, entry in sorted(registry.items())},
        "benchmarks": {
            key: {
                "records": value["records"],
                "psf_ready": value["psf_ready"],
                "variants": dict(value["variants"]),
                "models": dict(value["models"]),
                "multimodal_records": value.get("multimodal_records", 0),
            }
            for key, value in benchmark_counts.items()
        },
        "models": dict(model_counts),
        "issues": issues[:200],
        "notes": [
            "SquadA cleaned_data/ is the source of truth for active analysis.",
            "Aider is excluded from the default benchmark set per the latest instructions.",
        ],
    }
    dump_json(artifact_dir / "manifest.json", summary)
    return summary


def build_psf_outputs(
    repo_root: Path | None = None,
    scope: str = "group",
    bootstrap_iterations: int = 100,
    preserve_messages: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root or default_repo_root()
    artifact_dir = ensure_dir(_artifact_dir(repo_root))
    records, _ = load_all_records(repo_root=repo_root, preserve_messages=preserve_messages)
    psf_rows = build_psf_table(records, scope=scope, bootstrap_iterations=bootstrap_iterations)
    pairs = pairwise_same_pass1_pairs(psf_rows)

    dump_json(artifact_dir / f"psf_{scope}.json", psf_rows)
    dump_json(artifact_dir / f"psf_{scope}_pairs.json", pairs)
    return {"rows": psf_rows, "pairs": pairs}


def build_cluster_outputs(
    repo_root: Path | None = None,
    scope: str = "group",
    bootstrap_iterations: int = 100,
    k: int = 3,
    benchmark_controlled: bool = False,
    preserve_messages: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root or default_repo_root()
    artifact_dir = ensure_dir(_artifact_dir(repo_root))
    psf = build_psf_outputs(
        repo_root=repo_root,
        scope=scope,
        bootstrap_iterations=bootstrap_iterations,
        preserve_messages=preserve_messages,
    )
    if benchmark_controlled:
        clustering = cluster_psf_rows_by_benchmark(psf["rows"], k=k)
        artifact_name = f"clusters_k{k}_by_benchmark.json"
    else:
        clustering = cluster_psf_rows(psf["rows"], k=k)
        artifact_name = f"clusters_k{k}.json"
    dump_json(artifact_dir / artifact_name, clustering)
    return clustering


def build_cross_outputs(
    rpc_path: Path,
    repo_root: Path | None = None,
    scope: str = "group",
    bootstrap_iterations: int = 100,
    preserve_messages: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root or default_repo_root()
    artifact_dir = ensure_dir(_artifact_dir(repo_root))
    psf = build_psf_outputs(
        repo_root=repo_root,
        scope=scope,
        bootstrap_iterations=bootstrap_iterations,
        preserve_messages=preserve_messages,
    )
    rpc_rows = load_rpc_rows(rpc_path)
    cross = build_cross_analysis(psf["rows"], rpc_rows)
    dump_json(artifact_dir / "rpc_gamma_cross.json", cross)
    return cross


def build_all(
    repo_root: Path | None = None,
    scope: str = "group",
    bootstrap_iterations: int = 100,
    cluster_k: int = 3,
    benchmark_controlled: bool = False,
    preserve_messages: bool = False,
    rpc_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or default_repo_root()
    manifest = build_manifest(repo_root=repo_root, preserve_messages=preserve_messages)
    psf = build_psf_outputs(
        repo_root=repo_root,
        scope=scope,
        bootstrap_iterations=bootstrap_iterations,
        preserve_messages=preserve_messages,
    )
    if benchmark_controlled:
        clustering = cluster_psf_rows_by_benchmark(psf["rows"], k=cluster_k)
        cluster_artifact = f"clusters_k{cluster_k}_by_benchmark.json"
    else:
        clustering = cluster_psf_rows(psf["rows"], k=cluster_k)
        cluster_artifact = f"clusters_k{cluster_k}.json"
    artifact_dir = ensure_dir(_artifact_dir(repo_root))
    dump_json(artifact_dir / cluster_artifact, clustering)

    cross = None
    if rpc_path is not None:
        rpc_rows = load_rpc_rows(rpc_path)
        cross = build_cross_analysis(psf["rows"], rpc_rows)
        dump_json(artifact_dir / "rpc_gamma_cross.json", cross)

    return {"manifest": manifest, "psf": psf, "clustering": clustering, "cross": cross}
