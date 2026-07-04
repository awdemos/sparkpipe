#!/usr/bin/env python3
"""
Build one-time GLM-5.2 FP8 resident MoE packs from the official FP8 checkpoint.

The generated files are setup artifacts, not serving-path code.  Serving should
load these packs from C/CUDA only.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import struct
import tempfile
from typing import Any, BinaryIO, Dict, Iterable, List, Tuple



MAGIC = b"SPARKGLM52FP8\0\0"
ABI_VERSION = 1
HEADER_BYTES = 512
REGION_ALIGNMENT = 4096
REGION_FORMAT = "<QQ"
REGION_COUNT = 4
HIDDEN_DIMENSION = 6144
INTERMEDIATE_DIMENSION = 2048
EXPERT_COUNT = 256
TOP_K = 8
CUDA_ARCHITECTURE_SM121 = 121
GATE_UP_ORDER_UP_GATE = 1
WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR = 1
SCALE_LAYOUT_EXPERT_MAJOR_ROW_BLOCK_MAJOR = 1
QUANT_MODE_FP8_E4M3 = 2
OUTPUT_DTYPE_BF16 = 1
REGION_W1_WEIGHT = 0
REGION_W1_SCALE_INV = 1
REGION_W2_WEIGHT = 2
REGION_W2_SCALE_INV = 3


class PackFailure(RuntimeError):
    pass


def import_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise PackFailure("torch is required to build GLM-5.2 FP8 packs") from error
    return torch


def fp8_scale_inv_row_block_major_byte_count(rows: int, column_blocks: int) -> int:
    if rows <= 0 or column_blocks <= 0:
        raise PackFailure("FP8 scale dimensions must be positive")
    return rows * column_blocks * 4


def fp8_scale_inv_bytes_to_runtime_row_block_major(
    name: str,
    row_major: bytes,
    rows: int,
    column_blocks: int,
) -> bytes:
    expected_bytes = fp8_scale_inv_row_block_major_byte_count(rows, column_blocks)
    if len(row_major) != expected_bytes:
        raise PackFailure(
            f"{name} scale byte count {len(row_major)}, expected {expected_bytes} "
            f"for runtime shape ({rows}, {column_blocks})"
        )
    return row_major


def transposed_fp8_scale_inv_bytes_to_runtime_row_block_major(
    name: str,
    transposed: bytes,
    rows: int,
    column_blocks: int,
) -> bytes:
    expected_bytes = fp8_scale_inv_row_block_major_byte_count(rows, column_blocks)
    if len(transposed) != expected_bytes:
        raise PackFailure(
            f"{name} transposed scale byte count {len(transposed)}, "
            f"expected {expected_bytes} for checkpoint shape ({column_blocks}, {rows})"
        )
    runtime = bytearray(expected_bytes)
    for row in range(rows):
        for column_block in range(column_blocks):
            source = ((column_block * rows) + row) * 4
            target = ((row * column_blocks) + column_block) * 4
            runtime[target:target + 4] = transposed[source:source + 4]
    return bytes(runtime)


class SafetensorReader:
    def __init__(self, model_dir: Path) -> None:
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise PackFailure(
                "safetensors is required to build GLM-5.2 FP8 packs"
            ) from error
        index_path = model_dir / "model.safetensors.index.json"
        self.model_dir = model_dir
        self.safe_open = safe_open
        self.weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self.handles: Dict[str, Any] = {}

    def tensor(self, name: str) -> Any:
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"missing tensor in index: {name}")
        if shard not in self.handles:
            path = self.model_dir / shard
            if not path.exists():
                raise PackFailure(f"missing safetensors shard: {path}")
            self.handles[shard] = self.safe_open(str(path), framework="pt", device="cpu")
        handle = self.handles[shard]
        if name not in handle.keys():
            raise PackFailure(f"missing tensor in shard {shard}: {name}")
        return handle.get_tensor(name)

    def close(self) -> None:
        self.handles.clear()


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def tensor_name(layer: int, expert: int, projection: str, suffix: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.{suffix}"


def reserve_regions() -> List[Dict[str, int]]:
    w1_weight_bytes = EXPERT_COUNT * (2 * INTERMEDIATE_DIMENSION) * HIDDEN_DIMENSION
    w1_scale_bytes = EXPERT_COUNT * 32 * 48 * 4
    w2_weight_bytes = EXPERT_COUNT * HIDDEN_DIMENSION * INTERMEDIATE_DIMENSION
    w2_scale_bytes = EXPERT_COUNT * 48 * 16 * 4
    offset = HEADER_BYTES
    regions: List[Dict[str, int]] = []
    for byte_count in (w1_weight_bytes, w1_scale_bytes, w2_weight_bytes, w2_scale_bytes):
        offset = align_up(offset, REGION_ALIGNMENT)
        regions.append({"offset": offset, "bytes": byte_count})
        offset += byte_count
    return regions


def pack_header(layer: int, regions: List[Dict[str, int]]) -> bytes:
    prefix = struct.pack(
        "<16s16I",
        MAGIC,
        ABI_VERSION,
        HEADER_BYTES,
        layer,
        128,
        HIDDEN_DIMENSION,
        INTERMEDIATE_DIMENSION,
        EXPERT_COUNT,
        TOP_K,
        GATE_UP_ORDER_UP_GATE,
        WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR,
        SCALE_LAYOUT_EXPERT_MAJOR_ROW_BLOCK_MAJOR,
        QUANT_MODE_FP8_E4M3,
        OUTPUT_DTYPE_BF16,
        CUDA_ARCHITECTURE_SM121,
        0,
        0,
    )
    region_bytes = b"".join(
        struct.pack(REGION_FORMAT, region["offset"], region["bytes"])
        for region in regions
    )
    header = prefix + region_bytes
    if len(header) > HEADER_BYTES:
        raise PackFailure("FP8 pack header exceeds fixed header size")
    return header + (b"\0" * (HEADER_BYTES - len(header)))


def seek_region(file: BinaryIO, regions: List[Dict[str, int]], region_index: int) -> None:
    file.seek(regions[region_index]["offset"])


def write_tensor_bytes(file: BinaryIO, tensor: Any, expected_shape: Tuple[int, ...], name: str) -> None:
    torch = import_torch()
    if tuple(tensor.shape) != expected_shape:
        raise PackFailure(f"{name} has shape {tuple(tensor.shape)}, expected {expected_shape}")
    if "float8_e4m3" not in str(tensor.dtype):
        raise PackFailure(f"{name} has dtype {tensor.dtype}, expected FP8 E4M3")
    file.write(tensor.contiguous().view(torch.uint8).numpy().tobytes())


def fp8_scale_tensor_to_runtime_bytes(
    tensor: Any,
    expected_shape: Tuple[int, int],
    name: str,
    allow_transposed_scales: bool,
) -> bytes:
    shape = tuple(tensor.shape)
    if shape == expected_shape:
        source_tensor = tensor
    elif allow_transposed_scales and shape == (expected_shape[1], expected_shape[0]):
        source_tensor = tensor.transpose(0, 1)
    else:
        raise PackFailure(f"{name} has shape {shape}, expected {expected_shape}")
    if str(source_tensor.dtype) != "torch.float32":
        raise PackFailure(f"{name} has dtype {source_tensor.dtype}, expected float32")
    row_major = source_tensor.contiguous().numpy().astype("<f4", copy=False).tobytes()
    return fp8_scale_inv_bytes_to_runtime_row_block_major(
        name,
        row_major,
        expected_shape[0],
        expected_shape[1],
    )


def write_scale_bytes(
    file: BinaryIO,
    tensor: Any,
    expected_shape: Tuple[int, int],
    name: str,
    allow_transposed_scales: bool,
) -> None:
    file.write(
        fp8_scale_tensor_to_runtime_bytes(
            tensor,
            expected_shape,
            name,
            allow_transposed_scales,
        )
    )


def write_layer_pack(
    model_dir: Path,
    output_dir: Path,
    layer: int,
    reuse: bool,
    allow_transposed_scales: bool,
) -> Dict[str, Any]:
    reader = SafetensorReader(model_dir)
    regions = reserve_regions()
    output_path = output_dir / f"glm52_layer_{layer:04d}_fp8_moe.spfp8"
    expected_bytes = regions[-1]["offset"] + regions[-1]["bytes"]
    try:
        if reuse and output_path.exists() and output_path.stat().st_size == expected_bytes:
            return {"layer": layer, "path": str(output_path), "bytes": expected_bytes, "reused": True}
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".glm52_layer_{layer:04d}_",
            suffix=".tmp",
            dir=str(output_dir),
            delete=False,
        ) as file:
            tmp_path = Path(file.name)
            file.write(pack_header(layer, regions))
            seek_region(file, regions, REGION_W1_WEIGHT)
            for expert in range(EXPERT_COUNT):
                write_tensor_bytes(
                    file,
                    reader.tensor(tensor_name(layer, expert, "up_proj", "weight")),
                    (INTERMEDIATE_DIMENSION, HIDDEN_DIMENSION),
                    f"layer {layer} expert {expert} up weight",
                )
                write_tensor_bytes(
                    file,
                    reader.tensor(tensor_name(layer, expert, "gate_proj", "weight")),
                    (INTERMEDIATE_DIMENSION, HIDDEN_DIMENSION),
                    f"layer {layer} expert {expert} gate weight",
                )
            seek_region(file, regions, REGION_W1_SCALE_INV)
            for expert in range(EXPERT_COUNT):
                write_scale_bytes(
                    file,
                    reader.tensor(tensor_name(layer, expert, "up_proj", "weight_scale_inv")),
                    (16, 48),
                    f"layer {layer} expert {expert} up scale_inv",
                    allow_transposed_scales,
                )
                write_scale_bytes(
                    file,
                    reader.tensor(tensor_name(layer, expert, "gate_proj", "weight_scale_inv")),
                    (16, 48),
                    f"layer {layer} expert {expert} gate scale_inv",
                    allow_transposed_scales,
                )
            seek_region(file, regions, REGION_W2_WEIGHT)
            for expert in range(EXPERT_COUNT):
                write_tensor_bytes(
                    file,
                    reader.tensor(tensor_name(layer, expert, "down_proj", "weight")),
                    (HIDDEN_DIMENSION, INTERMEDIATE_DIMENSION),
                    f"layer {layer} expert {expert} down weight",
                )
            seek_region(file, regions, REGION_W2_SCALE_INV)
            for expert in range(EXPERT_COUNT):
                write_scale_bytes(
                    file,
                    reader.tensor(tensor_name(layer, expert, "down_proj", "weight_scale_inv")),
                    (48, 16),
                    f"layer {layer} expert {expert} down scale_inv",
                    allow_transposed_scales,
                )
        tmp_path.replace(output_path)
        return {"layer": layer, "path": str(output_path), "bytes": expected_bytes, "reused": False}
    finally:
        reader.close()


def parse_layers(value: str) -> List[int]:
    layers: List[int] = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if item:
            layers.append(int(item))
    return layers


def worker(argument: Tuple[str, str, int, bool, bool]) -> Dict[str, Any]:
    model_dir, output_dir, layer, reuse, allow_transposed_scales = argument
    return write_layer_pack(
        Path(model_dir),
        Path(output_dir),
        layer,
        reuse,
        allow_transposed_scales,
    )


def merge_manifest_records(output_dir: Path, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    manifest_path = output_dir / "fp8_moe_pack_manifest.json"
    merged: Dict[int, Dict[str, Any]] = {}
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        for record in prior.get("layers", []):
            merged[int(record["layer"])] = record
    for record in records:
        merged[int(record["layer"])] = record
    return [merged[layer] for layer in sorted(merged)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GLM-5.2 FP8 resident MoE packs")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--allow-transposed-scales", action="store_true")
    args = parser.parse_args()
    layers = parse_layers(args.layers)
    jobs = max(1, int(args.jobs))
    output_dir = Path(args.output_dir)
    tasks = [
        (
            args.model_dir,
            args.output_dir,
            layer,
            not args.no_reuse,
            bool(args.allow_transposed_scales),
        )
        for layer in layers
    ]
    if jobs == 1:
        records = [worker(task) for task in tasks]
    else:
        with Pool(processes=jobs) as pool:
            records = list(pool.imap_unordered(worker, tasks))
    records = sorted(records, key=lambda record: int(record["layer"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = merge_manifest_records(output_dir, records)
    (output_dir / "fp8_moe_pack_manifest.json").write_text(
        json.dumps(
            {
                "format": "sparkpipe.glm52.fp8.resident_moe_pack.v1",
                "model_dir": str(Path(args.model_dir)),
                "allow_transposed_scales": bool(args.allow_transposed_scales),
                "scale_layout": "expert_major_row_block_major_f32_scale_inv",
                "layers": manifest_records,
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    for record in records:
        print(
            "fp8_pack layer={layer} bytes={bytes} reused={reused} path={path}".format(
                **record
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
