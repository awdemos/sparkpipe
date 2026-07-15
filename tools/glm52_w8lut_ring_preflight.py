#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
from typing import Any


DEFAULT_HOSTS = (
    "spark0",
    "spark1",
    "spark2",
    "spark3",
    "spark4",
    "spark5",
    "spark6",
    "spark7",
    "spark8",
    "spark9",
    "sparka",
    "sparkb",
    "sparkc",
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
FIRST_ROUTED_LAYER = 3
LAYERS_PER_STAGE = 6


class RingPreflightFailure(RuntimeError):
    pass


def expand_path(template: str, host: str, rank: int) -> str:
    try:
        return template.format(host=host, rank=rank)
    except (KeyError, ValueError) as error:
        raise RingPreflightFailure(
            f"invalid path template {template!r}: {error}"
        ) from error


def run_remote(
    host: str,
    arguments: list[str],
    connect_timeout: int,
    command_timeout: int,
) -> subprocess.CompletedProcess[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        host,
        shlex.join(arguments),
    ]
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=command_timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RingPreflightFailure(
            f"{host}: command timed out after {command_timeout}s"
        ) from error


def require_remote_success(
    result: subprocess.CompletedProcess[str],
    host: str,
    operation: str,
) -> str:
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RingPreflightFailure(
            f"{host}: {operation} failed with status {result.returncode}: {detail}"
        )
    return result.stdout.strip()


def selected_layer(rank: int) -> int:
    return max(rank * LAYERS_PER_STAGE, FIRST_ROUTED_LAYER)


def validate_rank(
    rank: int,
    host: str,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    checkout = expand_path(arguments.checkout_root, host, rank)
    stagepack_root = expand_path(arguments.stagepack_root, host, rank)
    w8lut_root = expand_path(arguments.w8lut_pack_root, host, rank)
    try:
        head = require_remote_success(
            run_remote(
                host,
                ["git", "-C", checkout, "rev-parse", "HEAD"],
                arguments.connect_timeout,
                arguments.command_timeout,
            ),
            host,
            "read checkout commit",
        )
        if head != arguments.expected_commit:
            raise RingPreflightFailure(
                f"{host}: checkout commit {head} != {arguments.expected_commit}"
            )
        tracked_status = require_remote_success(
            run_remote(
                host,
                [
                    "git",
                    "-C",
                    checkout,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=no",
                ],
                arguments.connect_timeout,
                arguments.command_timeout,
            ),
            host,
            "read checkout status",
        )
        if tracked_status:
            raise RingPreflightFailure(
                f"{host}: checkout has tracked changes: {tracked_status.splitlines()[0]}"
            )
        preflight_arguments = [
            "python3",
            f"{checkout}/tools/glm52_w8lut_artifact_preflight.py",
            "--rank",
            str(rank),
            "--stagepack-root",
            stagepack_root,
            "--w8lut-pack-root",
            w8lut_root,
            "--sample-bytes",
            str(arguments.sample_bytes),
        ]
        if arguments.one_layer_per_rank:
            preflight_arguments.extend(["--layer", str(selected_layer(rank))])
        if arguments.verify_sha256:
            preflight_arguments.append("--verify-sha256")
        output = require_remote_success(
            run_remote(
                host,
                preflight_arguments,
                arguments.connect_timeout,
                arguments.command_timeout,
            ),
            host,
            "validate W8LUT artifacts",
        )
        try:
            artifact = json.loads(output)
        except json.JSONDecodeError as error:
            raise RingPreflightFailure(
                f"{host}: artifact preflight returned invalid JSON"
            ) from error
        if not isinstance(artifact, dict):
            raise RingPreflightFailure(
                f"{host}: artifact preflight receipt is not an object"
            )
        if artifact.get("status") != "ok" or artifact.get("rank") != rank:
            raise RingPreflightFailure(
                f"{host}: artifact receipt has the wrong status or rank"
            )
        return {
            "format": "sparkpipe.glm52.w8lut.ring_preflight.rank.v1",
            "status": "ok",
            "rank": rank,
            "host": host,
            "git_commit": head,
            "checkout": checkout,
            "artifact": artifact,
        }
    except (OSError, RingPreflightFailure) as error:
        return {
            "format": "sparkpipe.glm52.w8lut.ring_preflight.rank.v1",
            "status": "failed",
            "rank": rank,
            "host": host,
            "error": str(error),
        }


def parse_ranks(values: list[int] | None) -> list[int]:
    if values is None:
        return list(range(len(DEFAULT_HOSTS)))
    ranks = sorted(set(values))
    if not ranks or any(rank < 0 or rank >= len(DEFAULT_HOSTS) for rank in ranks):
        raise RingPreflightFailure("rank must be in 0..12")
    if len(ranks) != len(values):
        raise RingPreflightFailure("duplicate rank selection")
    return ranks


def parse_hosts(text: str) -> list[str]:
    hosts = [item.strip() for item in text.split(",") if item.strip()]
    if len(hosts) != len(DEFAULT_HOSTS) or len(set(hosts)) != len(hosts):
        raise RingPreflightFailure("--hosts must contain 13 unique host names")
    return hosts


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only zero-drift GLM-5.2 W8LUT artifact preflight across PP13",
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--hosts", default=",".join(DEFAULT_HOSTS))
    parser.add_argument(
        "--checkout-root",
        default="/home/{host}/src/sparkpipe-main-live",
    )
    parser.add_argument(
        "--stagepack-root",
        default=(
            "/home/{host}/sparkpipe_artifacts/"
            "glm52_w8lut_bf16_pp13_stage_payload_v1"
        ),
    )
    parser.add_argument(
        "--w8lut-pack-root",
        default=(
            "/home/{host}/sparkpipe_artifacts/"
            "glm52_w8lut_resident_moe_pp13_stage_v1"
        ),
    )
    parser.add_argument("--rank", action="append", type=int)
    parser.add_argument("--one-layer-per-rank", action="store_true")
    parser.add_argument("--sample-bytes", type=int, default=256)
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--command-timeout", type=int, default=1800)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def emit_record(
    record: dict[str, Any],
    receipt,
    lock: threading.Lock,
) -> None:
    line = json.dumps(record, sort_keys=True)
    with lock:
        print(line, flush=True)
        if receipt is not None:
            receipt.write(line + "\n")
            receipt.flush()


def main() -> int:
    arguments = parse_arguments()
    try:
        if COMMIT_RE.fullmatch(arguments.expected_commit) is None:
            raise RingPreflightFailure("--expected-commit must be a full lowercase SHA-1")
        hosts = parse_hosts(arguments.hosts)
        ranks = parse_ranks(arguments.rank)
        if arguments.sample_bytes < 16 or arguments.sample_bytes > 65536:
            raise RingPreflightFailure("--sample-bytes must be in 16..65536")
        if arguments.connect_timeout < 1 or arguments.command_timeout < 1:
            raise RingPreflightFailure("timeouts must be positive")
        if arguments.receipt is not None:
            arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt = arguments.receipt.open("x", encoding="utf-8")
        else:
            receipt = None
    except (OSError, RingPreflightFailure) as error:
        print(f"glm52_w8lut_ring_preflight: {error}", file=sys.stderr)
        return 2
    lock = threading.Lock()
    records: list[dict[str, Any]] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(ranks),
        ) as executor:
            futures = {
                executor.submit(
                    validate_rank,
                    rank,
                    hosts[rank],
                    arguments,
                ): rank
                for rank in ranks
            }
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                records.append(record)
                emit_record(record, receipt, lock)
        failed = sorted(
            record["rank"]
            for record in records
            if record["status"] != "ok"
        )
        summary = {
            "format": "sparkpipe.glm52.w8lut.ring_preflight.summary.v1",
            "status": "ok" if not failed else "failed",
            "git_commit": arguments.expected_commit,
            "rank_count": len(records),
            "passed_rank_count": len(records) - len(failed),
            "failed_ranks": failed,
            "scope": (
                "one-layer-per-rank"
                if arguments.one_layer_per_rank
                else "full-stage"
            ),
        }
        emit_record(summary, receipt, lock)
        return 0 if not failed else 1
    finally:
        if receipt is not None:
            receipt.close()


if __name__ == "__main__":
    raise SystemExit(main())
