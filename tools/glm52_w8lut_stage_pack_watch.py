#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import glm52_stage_pack as stage_pack
from glm52_model_contract import load_model_contract
from glm52_resident_pack_common import EXPERT_COUNT, parse_layers, tensor_name


EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a rank's exact BF16 source shards, then build its "
            "BF16 StagePack and W8LUT expert packs"
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--packer", required=True, type=Path)
    parser.add_argument("--stage-packer", type=Path)
    parser.add_argument("--stage-output-dir", type=Path)
    parser.add_argument(
        "--required-files-mode",
        choices=("pack-exact", "manifest"),
        default="pack-exact",
    )
    parser.add_argument("--max-active", type=int, default=1024)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=0.0)
    return parser.parse_args()


def manifest_required_files(manifest: dict, rank: int) -> list[str]:
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


def expected_expert_layers(rank: int) -> list[int]:
    contract = load_model_contract()
    if rank < 0 or rank >= stage_pack.STAGE_COUNT:
        raise ValueError(f"rank {rank} is outside PP{stage_pack.STAGE_COUNT}")
    first_layer = rank * stage_pack.LAYERS_PER_STAGE
    end_layer = min(
        first_layer + stage_pack.LAYERS_PER_STAGE,
        contract["layer_count"],
    )
    layers = list(range(
        max(first_layer, contract["first_routed_layer"]),
        end_layer,
    ))
    if rank == stage_pack.STAGE_COUNT - 1:
        layers.append(contract["layer_count"])
    return layers


def selected_expert_layers(rank: int, text: str) -> list[int]:
    if text == "auto":
        return expected_expert_layers(rank)
    return parse_layers(text)


def exact_pack_required_files(
    model_dir: Path,
    rank: int,
    layers: list[int],
    include_stage_pack: bool,
) -> list[str]:
    weight_map = stage_pack.load_weight_map(model_dir)
    tensor_names: set[str] = set()
    if include_stage_pack:
        stage_tensors, _ = stage_pack.collect_stage_tensors(weight_map, [rank])
        tensor_names.update(
            stage_pack.source_tensor_name(name)
            for name in stage_tensors[rank]
        )
    for layer in layers:
        for expert in range(EXPERT_COUNT):
            for projection in EXPERT_PROJECTIONS:
                name = tensor_name(layer, expert, projection, "weight")
                if name not in weight_map:
                    raise ValueError(f"BF16 index is missing W8LUT source tensor: {name}")
                tensor_names.add(name)
    files = {"model.safetensors.index.json"}
    files.update(str(weight_map[name]) for name in tensor_names)
    return sorted(files)


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


def build_stage_pack(args: argparse.Namespace) -> None:
    if args.stage_packer is None and args.stage_output_dir is None:
        return
    if args.stage_packer is None or args.stage_output_dir is None:
        raise ValueError(
            "--stage-packer and --stage-output-dir must be provided together"
        )
    command = [
        sys.executable,
        str(args.stage_packer),
        "--model-dir", str(args.model_dir),
        "--output-dir", str(args.stage_output_dir),
        "--model-quantization", stage_pack.MODEL_QUANTIZATION_W8LUT,
        "--stages", str(args.rank),
        "--reuse",
    ]
    print(
        f"w8lut_stage_non_expert_pack_start rank={args.rank}",
        flush=True,
    )
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    layers = selected_expert_layers(args.rank, args.layers)
    if args.required_files_mode == "manifest":
        files = manifest_required_files(manifest, args.rank)
    else:
        manifest_required_files(manifest, args.rank)
        files = exact_pack_required_files(
            args.model_dir,
            args.rank,
            layers,
            args.stage_packer is not None,
        )
    wait_for_files(
        args.model_dir, files, args.poll_seconds, args.wait_timeout_seconds)
    build_stage_pack(args)
    command = [
        sys.executable,
        str(args.packer),
        "--model-dir", str(args.model_dir),
        "--output-dir", str(args.output_dir),
        "--layers", ",".join(str(layer) for layer in layers),
        "--jobs", str(args.jobs),
        "--max-active", str(args.max_active),
    ]
    print(
        f"w8lut_stage_pack_start rank={args.rank} "
        f"layers={','.join(str(layer) for layer in layers)}",
        flush=True,
    )
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
