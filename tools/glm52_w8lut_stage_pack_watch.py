#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for a rank's waterfall files, then build its W8LUT packs")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--packer", required=True, type=Path)
    parser.add_argument("--max-active", type=int, default=1024)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=0.0)
    return parser.parse_args()


def required_files(manifest: dict, rank: int) -> list[str]:
    nodes = manifest.get("nodes", [])
    if rank < 0 or rank >= len(nodes):
        raise ValueError(f"rank {rank} is outside manifest node count {len(nodes)}")
    files = []
    for item in manifest.get("files", []):
        if rank in item.get("needed_ranks", []):
            files.append(str(item["rel"]))
    if not files:
        raise ValueError(f"manifest assigns no files to rank {rank}")
    return files


def wait_for_files(model_dir: Path, files: list[str], poll_seconds: float,
                   timeout_seconds: float) -> None:
    started = time.monotonic()
    previous_count = -1
    while True:
        missing = [name for name in files if not (model_dir / name).is_file()]
        if not missing:
            print(f"w8lut_stage_source_ready files={len(files)}", flush=True)
            return
        if len(missing) != previous_count:
            print(
                f"w8lut_stage_wait missing={len(missing)} next={missing[0]}",
                flush=True)
            previous_count = len(missing)
        if timeout_seconds > 0 and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"timed out waiting for {len(missing)} files")
        time.sleep(poll_seconds)


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    files = required_files(manifest, args.rank)
    wait_for_files(
        args.model_dir, files, args.poll_seconds, args.wait_timeout_seconds)
    command = [
        sys.executable,
        str(args.packer),
        "--model-dir", str(args.model_dir),
        "--output-dir", str(args.output_dir),
        "--layers", args.layers,
        "--jobs", str(args.jobs),
        "--max-active", str(args.max_active),
    ]
    print(f"w8lut_stage_pack_start rank={args.rank} layers={args.layers}", flush=True)
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
