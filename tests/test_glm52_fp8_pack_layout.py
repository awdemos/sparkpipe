#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import sys
import tempfile


def load_fp8_packer_module():
    repo_root = Path(__file__).resolve().parents[1]
    packer_path = repo_root / "tools" / "glm52_fp8_resident_pack.py"
    sys.path.insert(0, str(packer_path.parent))
    spec = importlib.util.spec_from_file_location("glm52_fp8_resident_pack", packer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load FP8 packer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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
    repo_root = Path(__file__).resolve().parents[1]
    cuda_source = (
        repo_root /
        "modules/glm52_resident_decode_stage/source/spark_glm52_sm121_required_decode_stage.cu"
    ).read_text(encoding="utf-8")
    assert "SPARKPIPE_FP8_MOE_ACCURATE_BF16_ACTIVATION" not in cuda_source
    assert "Fp8MoeBf16ActivationScalarGroupGemm" not in cuda_source
    assert "__nv_cvt_float_to_fp8" in cuda_source
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
    regions = module.reserve_regions()
    header = module.pack_header(7, regions, 1024)
    header_fields, header_regions = module.unpack_pack_header(header)
    assert header_fields.layer_index == 7
    assert header_fields.maximum_token_count == 1024
    assert header_regions == regions
    assert module.DEFAULT_MAX_ACTIVE_SEQUENCE_COUNT == 1024
    with tempfile.TemporaryDirectory() as temp_dir:
        pack_path = Path(temp_dir) / "glm52_layer_0007_fp8_moe.spfp8"
        pack_bytes = regions[-1]["offset"] + regions[-1]["bytes"]
        with pack_path.open("wb") as file:
            file.write(module.pack_header(7, regions, 128))
            file.truncate(pack_bytes)
        assert module.existing_pack_can_reuse(pack_path, 7, pack_bytes, 1024)
        updated_header, updated_regions = module.existing_pack_header(pack_path)
        assert updated_header.maximum_token_count == 1024
        assert updated_regions == regions
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
