#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, BinaryIO, Dict, List, Tuple

import numpy as np

from glm52_resident_pack_common import (
    EXPERT_COUNT,
    HIDDEN_DIMENSION,
    INTERMEDIATE_DIMENSION,
    TOP_K,
    W1_COMPONENT_COUNT,
    PackFailure,
    SafetensorReader,
    align_up,
    import_torch,
    parse_layers,
    tensor_name,
)
import glm52_w8lut_codec as w8lut


MAGIC = b"SPARKGLM52W8LUT"
MAGIC_FIELD_BYTES = 16
WIRE_MAGIC = MAGIC.ljust(MAGIC_FIELD_BYTES, b"\0")
ABI_VERSION = 1
HEADER_BYTES = 512
REGION_ALIGNMENT = 4096
HEADER_U32_FIELD_COUNT = 16
HEADER_PREFIX_STRUCT = struct.Struct(f"<{MAGIC_FIELD_BYTES}s{HEADER_U32_FIELD_COUNT}I")
REGION_STRUCT = struct.Struct("<QQ")
REGION_COUNT = 4
UINT16_BYTES = struct.calcsize("<H")
CUDA_ARCHITECTURE_SM121 = 121
GATE_UP_ORDER_UP_GATE = 1
WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR = 1
SCALE_LAYOUT_W8LUT_EXPERT_COMPONENT_E0 = 2
QUANT_MODE_W8LUT = 3
OUTPUT_DTYPE_BF16 = 1
DEFAULT_MAX_ACTIVE_SEQUENCE_COUNT = 1024
REGION_W1_CODES = 0
REGION_W1_E0 = 1
REGION_W2_CODES = 2
REGION_W2_E0 = 3
PACK_FILE_TEMPLATE = "glm52_layer_{layer:04d}_w8lut_moe.spw8lut"
MANIFEST_FILE = "w8lut_moe_pack_manifest.json"
HASH_CHUNK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class W8lutMoePackHeader:
    magic: bytes
    abi_version: int
    header_bytes: int
    layer_index: int
    maximum_token_count: int
    hidden_dimension: int
    intermediate_dimension: int
    expert_count: int
    top_k: int
    gate_up_order: int
    weight_layout: int
    scale_layout: int
    quant_mode: int
    output_dtype: int
    cuda_architecture: int
    reserved0: int
    reserved1: int


def reserve_regions() -> List[Dict[str, int]]:
    w1_rows = W1_COMPONENT_COUNT * INTERMEDIATE_DIMENSION
    byte_counts = (
        EXPERT_COUNT * w1_rows * HIDDEN_DIMENSION,
        EXPERT_COUNT * W1_COMPONENT_COUNT * UINT16_BYTES,
        EXPERT_COUNT * HIDDEN_DIMENSION * INTERMEDIATE_DIMENSION,
        EXPERT_COUNT * UINT16_BYTES,
    )
    offset = HEADER_BYTES
    regions: List[Dict[str, int]] = []
    for byte_count in byte_counts:
        offset = align_up(offset, REGION_ALIGNMENT)
        regions.append({"offset": offset, "bytes": byte_count})
        offset += byte_count
    return regions


def expected_pack_header(layer: int, max_active: int) -> W8lutMoePackHeader:
    if max_active <= 0:
        raise PackFailure("maximum active sequence count must be positive")
    return W8lutMoePackHeader(
        magic=WIRE_MAGIC,
        abi_version=ABI_VERSION,
        header_bytes=HEADER_BYTES,
        layer_index=layer,
        maximum_token_count=max_active,
        hidden_dimension=HIDDEN_DIMENSION,
        intermediate_dimension=INTERMEDIATE_DIMENSION,
        expert_count=EXPERT_COUNT,
        top_k=TOP_K,
        gate_up_order=GATE_UP_ORDER_UP_GATE,
        weight_layout=WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR,
        scale_layout=SCALE_LAYOUT_W8LUT_EXPERT_COMPONENT_E0,
        quant_mode=QUANT_MODE_W8LUT,
        output_dtype=OUTPUT_DTYPE_BF16,
        cuda_architecture=CUDA_ARCHITECTURE_SM121,
        reserved0=0,
        reserved1=0,
    )


def pack_header(layer: int, regions: List[Dict[str, int]], max_active: int) -> bytes:
    fields = expected_pack_header(layer, max_active)
    prefix = HEADER_PREFIX_STRUCT.pack(
        fields.magic,
        fields.abi_version,
        fields.header_bytes,
        fields.layer_index,
        fields.maximum_token_count,
        fields.hidden_dimension,
        fields.intermediate_dimension,
        fields.expert_count,
        fields.top_k,
        fields.gate_up_order,
        fields.weight_layout,
        fields.scale_layout,
        fields.quant_mode,
        fields.output_dtype,
        fields.cuda_architecture,
        fields.reserved0,
        fields.reserved1,
    )
    region_bytes = b"".join(
        REGION_STRUCT.pack(region["offset"], region["bytes"])
        for region in regions
    )
    header = prefix + region_bytes
    if len(header) > HEADER_BYTES:
        raise PackFailure("W8LUT pack header exceeds fixed header size")
    return header + (b"\0" * (HEADER_BYTES - len(header)))


def unpack_pack_header(header: bytes) -> Tuple[W8lutMoePackHeader, List[Dict[str, int]]]:
    if len(header) != HEADER_BYTES:
        raise PackFailure("short W8LUT pack header")
    fields = W8lutMoePackHeader(*HEADER_PREFIX_STRUCT.unpack(header[:HEADER_PREFIX_STRUCT.size]))
    regions = []
    offset = HEADER_PREFIX_STRUCT.size
    for _ in range(REGION_COUNT):
        region_offset, region_bytes = REGION_STRUCT.unpack(header[offset:offset + REGION_STRUCT.size])
        regions.append({"offset": region_offset, "bytes": region_bytes})
        offset += REGION_STRUCT.size
    return fields, regions


def seek_region(file: BinaryIO, regions: List[Dict[str, int]], region_index: int) -> None:
    file.seek(regions[region_index]["offset"])


def tensor_bf16_bits(tensor: Any, expected_shape: Tuple[int, ...], name: str):
    torch = import_torch()
    if tuple(tensor.shape) != expected_shape:
        raise PackFailure(f"{name} has shape {tuple(tensor.shape)}, expected {expected_shape}")
    if tensor.dtype != torch.bfloat16:
        raise PackFailure(f"{name} has dtype {tensor.dtype}, expected torch.bfloat16")
    signed_bits = tensor.contiguous().view(torch.int16).numpy()
    return signed_bits.view(np.uint16).reshape(-1)


def encode_tensor(tensor: Any, expected_shape: Tuple[int, ...], name: str):
    source = tensor_bf16_bits(tensor, expected_shape, name)
    try:
        codes, e0, stats = w8lut.encode(source)
        w8lut.verify(source, codes, e0)
    except ValueError as error:
        raise PackFailure(f"{name}: {error}") from error
    return codes, int(e0), stats


def write_u16_values(file: BinaryIO, values: List[int]) -> None:
    file.write(struct.pack(f"<{len(values)}H", *values))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_directory(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise PackFailure(f"W8LUT output path is not a directory: {output_dir}")
    for path in output_dir.iterdir():
        is_pack = path.is_file() and path.suffix == ".spw8lut"
        is_manifest = path.is_file() and path.name == MANIFEST_FILE
        is_pack_temp = path.is_file() and path.name.startswith(".glm52_layer_") and \
            "_w8lut_" in path.name and path.suffix == ".tmp"
        is_manifest_temp = path.is_file() and path.name.startswith(".w8lut_manifest_") and \
            path.suffix == ".tmp"
        if not (is_pack or is_manifest or is_pack_temp or is_manifest_temp):
            raise PackFailure(f"W8LUT output directory contains non-W8LUT artifact: {path}")


def write_layer_pack(model_dir: Path, output_dir: Path, layer: int, max_active: int) -> Dict[str, Any]:
    validate_output_directory(output_dir)
    output_path = output_dir / PACK_FILE_TEMPLATE.format(layer=layer)
    if output_path.exists():
        raise PackFailure(f"refusing to replace existing W8LUT pack: {output_path}")
    reader = SafetensorReader(model_dir)
    regions = reserve_regions()
    expected_bytes = regions[-1]["offset"] + regions[-1]["bytes"]
    w1_e0: List[int] = []
    w2_e0: List[int] = []
    below_window_count = 0
    element_count = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".glm52_layer_{layer:04d}_w8lut_",
            suffix=".tmp",
            dir=str(output_dir),
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            file.write(pack_header(layer, regions, max_active))
            seek_region(file, regions, REGION_W1_CODES)
            for expert in range(EXPERT_COUNT):
                for projection in ("up_proj", "gate_proj"):
                    name = tensor_name(layer, expert, projection, "weight")
                    codes, e0, stats = encode_tensor(
                        reader.tensor(name),
                        (INTERMEDIATE_DIMENSION, HIDDEN_DIMENSION),
                        name,
                    )
                    file.write(codes.tobytes(order="C"))
                    w1_e0.append(e0)
                    below_window_count += stats["below_window_count"]
                    element_count += stats["element_count"]
            seek_region(file, regions, REGION_W1_E0)
            write_u16_values(file, w1_e0)
            seek_region(file, regions, REGION_W2_CODES)
            for expert in range(EXPERT_COUNT):
                name = tensor_name(layer, expert, "down_proj", "weight")
                codes, e0, stats = encode_tensor(
                    reader.tensor(name),
                    (HIDDEN_DIMENSION, INTERMEDIATE_DIMENSION),
                    name,
                )
                file.write(codes.tobytes(order="C"))
                w2_e0.append(e0)
                below_window_count += stats["below_window_count"]
                element_count += stats["element_count"]
            seek_region(file, regions, REGION_W2_E0)
            write_u16_values(file, w2_e0)
            if file.tell() != expected_bytes:
                raise PackFailure(f"W8LUT pack byte count {file.tell()}, expected {expected_bytes}")
            file.flush()
            os.fsync(file.fileno())
        digest = sha256_file(temp_path)
        os.replace(temp_path, output_path)
        temp_path = None
        return {
            "layer": layer,
            "file": output_path.name,
            "path": str(output_path),
            "bytes": expected_bytes,
            "sha256": digest,
            "maximum_token_count": max_active,
            "element_count": element_count,
            "below_window_count": below_window_count,
            "below_window_ppm": below_window_count * 1000000 // element_count,
        }
    finally:
        reader.close()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def worker(argument: Tuple[str, str, int, int]) -> Dict[str, Any]:
    model_dir, output_dir, layer, max_active = argument
    return write_layer_pack(Path(model_dir), Path(output_dir), layer, max_active)


def index_sha256(model_dir: Path) -> str:
    return sha256_file(model_dir / "model.safetensors.index.json")


def write_manifest(output_dir: Path, model_dir: Path, max_active: int, records: List[Dict[str, Any]]) -> None:
    validate_output_directory(output_dir)
    manifest_path = output_dir / MANIFEST_FILE
    if manifest_path.exists():
        raise PackFailure(f"refusing to replace existing W8LUT manifest: {manifest_path}")
    manifest = {
        "format": "sparkpipe.glm52.w8lut.resident_moe_pack.v1",
        "pack_magic": WIRE_MAGIC.rstrip(b"\0").decode("ascii"),
        "pack_extension": ".spw8lut",
        "source_model_dir": str(model_dir.resolve()),
        "source_model_index_sha256": index_sha256(model_dir),
        "source_dtype": "BF16",
        "maximum_active_sequence_count": max_active,
        "quant_mode": QUANT_MODE_W8LUT,
        "scale_layout": "expert_component_u16_e0",
        "layers": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".w8lut_manifest_",
            suffix=".tmp",
            dir=str(output_dir),
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, manifest_path)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GLM-5.2 W8LUT resident MoE packs from BF16")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-active", type=int, default=DEFAULT_MAX_ACTIVE_SEQUENCE_COUNT)
    args = parser.parse_args()
    layers = parse_layers(args.layers)
    if args.max_active <= 0:
        raise PackFailure("--max-active must be positive")
    tasks = [
        (str(args.model_dir), str(args.output_dir), layer, args.max_active)
        for layer in layers
    ]
    jobs = max(1, min(int(args.jobs), len(tasks)))
    if jobs == 1:
        records = [worker(task) for task in tasks]
    else:
        with Pool(processes=jobs) as pool:
            records = list(pool.imap_unordered(worker, tasks))
    records.sort(key=lambda record: int(record["layer"]))
    write_manifest(args.output_dir, args.model_dir, args.max_active, records)
    for record in records:
        print(
            "w8lut_pack layer={layer} bytes={bytes} below_ppm={below_window_ppm} "
            "sha256={sha256} path={path}".format(**record)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
