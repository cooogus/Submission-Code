from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .pipeline import build_all, build_cluster_outputs, build_cross_outputs, build_manifest, build_psf_outputs
from .registry import default_repo_root


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=default_repo_root(), help="Repository root that contains outputs/")
    parser.add_argument("--scope", choices=["group", "variant"], default="group", help="Whether to aggregate at suite or variant level.")
    parser.add_argument("--bootstrap-iterations", type=int, default=100, help="Bootstrap samples for PSF confidence intervals.")
    parser.add_argument("--preserve-messages", action="store_true", help="Store normalized reasoning text in the loaded records.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="squadc", description="Squad C analysis pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Scan the repo and summarize what data is present.")
    _add_common_args(manifest)

    psf = subparsers.add_parser("psf", help="Fit the Pass@k fingerprint model and write artifacts.")
    _add_common_args(psf)

    cluster = subparsers.add_parser("cluster", help="Cluster PSF fingerprints in (alpha, beta, gamma) space.")
    _add_common_args(cluster)
    cluster.add_argument("--k", type=int, default=3, help="Number of clusters.")
    cluster.add_argument(
        "--benchmark-controlled",
        action="store_true",
        help="Cluster PSF rows separately within each benchmark before summarizing families.",
    )

    cross = subparsers.add_parser("cross", help="Join PSF rows with an RPC table and compute gamma correlations.")
    _add_common_args(cross)
    cross.add_argument("--rpc-path", type=Path, required=True, help="Path to an RPC summary table in JSON or CSV form.")

    all_cmd = subparsers.add_parser("all", help="Run manifest, PSF, clustering, and optional cross-analysis.")
    _add_common_args(all_cmd)
    all_cmd.add_argument("--k", type=int, default=3, help="Number of clusters.")
    all_cmd.add_argument(
        "--benchmark-controlled",
        action="store_true",
        help="Cluster PSF rows separately within each benchmark before summarizing families.",
    )
    all_cmd.add_argument("--rpc-path", type=Path, default=None, help="Optional RPC summary table path.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "manifest":
        build_manifest(repo_root=args.repo_root, preserve_messages=args.preserve_messages)
        return 0
    if args.command == "psf":
        build_psf_outputs(
            repo_root=args.repo_root,
            scope=args.scope,
            bootstrap_iterations=args.bootstrap_iterations,
            preserve_messages=args.preserve_messages,
        )
        return 0
    if args.command == "cluster":
        build_cluster_outputs(
            repo_root=args.repo_root,
            scope=args.scope,
            bootstrap_iterations=args.bootstrap_iterations,
            k=args.k,
            benchmark_controlled=args.benchmark_controlled,
            preserve_messages=args.preserve_messages,
        )
        return 0
    if args.command == "cross":
        build_cross_outputs(
            rpc_path=args.rpc_path,
            repo_root=args.repo_root,
            scope=args.scope,
            bootstrap_iterations=args.bootstrap_iterations,
            preserve_messages=args.preserve_messages,
        )
        return 0
    if args.command == "all":
        build_all(
            repo_root=args.repo_root,
            scope=args.scope,
            bootstrap_iterations=args.bootstrap_iterations,
            cluster_k=args.k,
            benchmark_controlled=args.benchmark_controlled,
            preserve_messages=args.preserve_messages,
            rpc_path=args.rpc_path,
        )
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2
