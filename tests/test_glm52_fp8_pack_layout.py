#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import struct


def load_fp8_packer_module():
    repo_root = Path(__file__).resolve().parents[1]
    packer_path = repo_root / "tools" / "glm52_fp8_resident_pack.py"
    spec = importlib.util.spec_from_file_location("glm52_fp8_resident_pack", packer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load FP8 packer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pack_f32_values(values):
    return b"".join(struct.pack("<f", float(value)) for value in values)


def unpack_f32_values(payload):
    return [
        struct.unpack("<f", payload[offset:offset + 4])[0]
        for offset in range(0, len(payload), 4)
    ]


def row_major_values(rows: int, column_blocks: int):
    return [
        (row * 100.0) + float(column_block)
        for row in range(rows)
        for column_block in range(column_blocks)
    ]


def transposed_values(rows: int, column_blocks: int):
    return [
        (row * 100.0) + float(column_block)
        for column_block in range(column_blocks)
        for row in range(rows)
    ]


def main() -> int:
    module = load_fp8_packer_module()
    assert module.ABI_VERSION == 1
    assert module.SCALE_LAYOUT_EXPERT_MAJOR_ROW_BLOCK_MAJOR == 1
    runtime_values = row_major_values(3, 4)
    runtime_bytes = pack_f32_values(runtime_values)
    assert module.fp8_scale_inv_bytes_to_runtime_row_block_major(
        "runtime scale",
        runtime_bytes,
        3,
        4,
    ) == runtime_bytes
    converted = module.transposed_fp8_scale_inv_bytes_to_runtime_row_block_major(
        "transposed scale",
        pack_f32_values(transposed_values(3, 4)),
        3,
        4,
    )
    assert unpack_f32_values(converted) == runtime_values
    try:
        module.fp8_scale_inv_bytes_to_runtime_row_block_major(
            "short scale",
            runtime_bytes[:-4],
            3,
            4,
        )
    except module.PackFailure:
        pass
    else:
        raise AssertionError("short FP8 scale layout was accepted")
    try:
        module.transposed_fp8_scale_inv_bytes_to_runtime_row_block_major(
            "short transposed scale",
            runtime_bytes[:-4],
            3,
            4,
        )
    except module.PackFailure:
        pass
    else:
        raise AssertionError("short transposed FP8 scale layout was accepted")
    assert module.parse_layers("3,4;5") == [3, 4, 5]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
