#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def load_tool(repository: Path):
    path = repository / "tools" / "glm52_w8lut_artifact_preflight.py"
    sys.path.insert(0, str(path.parent))
    specification = importlib.util.spec_from_file_location(
        "glm52_w8lut_artifact_preflight",
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load W8LUT artifact preflight")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def write_at(file_descriptor: int, offset: int, payload: bytes) -> None:
    written = os.pwrite(file_descriptor, payload, offset)
    if written != len(payload):
        raise RuntimeError("short sparse fixture write")


def create_stagepack(tool, root: Path, rank: int, layer: int, source_hash: str) -> None:
    file_name = f"stage_{rank:02d}_non_moe.spstage"
    path = root / file_name
    tensor_map = {}
    offset = 0
    selected = set(tool.required_layer_tensors(layer))
    for name, (dtype, shape) in sorted(tool.required_stage_tensors(rank).items()):
        offset = tool.align_up(offset, tool.STAGE_REGION_ALIGNMENT)
        byte_count = tool.tensor_bytes(dtype, shape)
        tensor_map[name] = {
            "file": file_name,
            "offset": offset,
            "bytes": byte_count,
            "dtype": dtype,
            "shape": list(shape),
            "source_shard": "fixture.safetensors",
        }
        offset += byte_count
    file_descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(file_descriptor, offset)
        for name in selected:
            record = tensor_map[name]
            byte_count = record["bytes"]
            width = min(byte_count, 64)
            for sample_index, relative_offset in enumerate(
                tool.sample_offsets(byte_count, 64)
            ):
                payload = bytes(
                    ((value + sample_index + len(name)) % 251) + 1
                    for value in range(width)
                )
                write_at(
                    file_descriptor,
                    record["offset"] + relative_offset,
                    payload,
                )
    finally:
        os.close(file_descriptor)
    index = {
        "format": tool.STAGEPACK_FORMAT,
        "topology": tool.STAGEPACK_TOPOLOGY,
        "model_quantization": tool.STAGEPACK_QUANTIZATION,
        "non_expert_weight_dtype": tool.STAGEPACK_DTYPE,
        "stage_count": tool.STAGE_COUNT,
        "layers_per_stage": tool.LAYERS_PER_STAGE,
        "source_model_index_sha256": source_hash,
        "stages": {
            str(rank): {
                "file": file_name,
                "first_layer": rank * tool.LAYERS_PER_STAGE,
                "layer_count": tool.LAYERS_PER_STAGE,
                "tensor_count": len(tensor_map),
            },
        },
        "tensor_map": tensor_map,
    }
    (root / tool.STAGEPACK_INDEX).write_text(
        json.dumps(index),
        encoding="utf-8",
    )


def write_code_samples(
    tool,
    file_descriptor: int,
    region_offset: int,
    route_count: int,
    route_bytes: int,
) -> None:
    for route in range(route_count):
        for sample_index, relative_offset in enumerate(
            tool.sample_offsets(route_bytes, 64)
        ):
            payload = bytes(
                (value + route + (sample_index * 67)) % 256
                for value in range(64)
            )
            write_at(
                file_descriptor,
                region_offset + (route * route_bytes) + relative_offset,
                payload,
            )


def create_w8lut_pack(
    tool,
    root: Path,
    layer: int,
    source_hash: str,
) -> Path:
    maximum_tokens = 1024
    file_name = f"glm52_layer_{layer:04d}_w8lut_moe.spw8lut"
    path = root / file_name
    regions = tool.expected_pack_regions()
    file_bytes = regions[-1][0] + regions[-1][1]
    file_descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(file_descriptor, file_bytes)
        write_at(
            file_descriptor,
            0,
            tool.build_pack_header(layer, maximum_tokens),
        )
        write_at(
            file_descriptor,
            regions[1][0],
            (120).to_bytes(2, "little") *
            (tool.EXPERT_COUNT * tool.W1_COMPONENT_COUNT),
        )
        write_at(
            file_descriptor,
            regions[3][0],
            (121).to_bytes(2, "little") * tool.EXPERT_COUNT,
        )
        route_bytes = tool.MOE_INTERMEDIATE_DIMENSION * tool.HIDDEN_DIMENSION
        write_code_samples(
            tool,
            file_descriptor,
            regions[0][0],
            tool.EXPERT_COUNT * tool.W1_COMPONENT_COUNT,
            route_bytes,
        )
        write_code_samples(
            tool,
            file_descriptor,
            regions[2][0],
            tool.EXPERT_COUNT,
            route_bytes,
        )
    finally:
        os.close(file_descriptor)
    element_count = regions[0][1] + regions[2][1]
    manifest = {
        "format": tool.W8LUT_FORMAT,
        "pack_magic": tool.W8LUT_MAGIC.decode("ascii"),
        "pack_extension": tool.W8LUT_EXTENSION,
        "source_model_index_sha256": source_hash,
        "source_dtype": tool.STAGEPACK_DTYPE,
        "maximum_active_sequence_count": maximum_tokens,
        "quant_mode": tool.W8LUT_QUANT_MODE,
        "scale_layout": "expert_component_u16_e0",
        "layers": [{
            "layer": layer,
            "file": file_name,
            "bytes": file_bytes,
            "sha256": "b" * 64,
            "maximum_token_count": maximum_tokens,
            "element_count": element_count,
            "below_window_count": 0,
            "below_window_ppm": 0,
        }],
    }
    (root / tool.W8LUT_MANIFEST).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return path


def run_preflight(
    script: Path,
    stagepack_root: Path,
    w8lut_root: Path,
    rank: int,
    layer: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(script),
            "--rank",
            str(rank),
            "--layer",
            str(layer),
            "--stagepack-root",
            str(stagepack_root),
            "--w8lut-pack-root",
            str(w8lut_root),
            "--sample-bytes",
            "64",
        ],
        capture_output=True,
        text=True,
    )


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "tools" / "glm52_w8lut_artifact_preflight.py"
    tool = load_tool(repository)
    rank = 2
    layer = 12
    source_hash = "a" * 64
    with tempfile.TemporaryDirectory(
        prefix="sparkpipe_w8lut_artifact_preflight_",
    ) as directory:
        root = Path(directory)
        stagepack_root = root / "stagepack"
        w8lut_root = root / "w8lut"
        stagepack_root.mkdir()
        w8lut_root.mkdir()
        create_stagepack(tool, stagepack_root, rank, layer, source_hash)
        pack_path = create_w8lut_pack(tool, w8lut_root, layer, source_hash)
        result = run_preflight(
            script,
            stagepack_root,
            w8lut_root,
            rank,
            layer,
        )
        assert result.returncode == 0, result.stderr
        receipt = json.loads(result.stdout)
        assert receipt["status"] == "ok"
        assert receipt["rank"] == rank
        assert receipt["scope"] == "selected-layers"
        assert [item["layer"] for item in receipt["layers"]] == [layer]
        assert receipt["layers"][0]["sample_unique_codes"] == 256
        manifest_path = w8lut_root / tool.W8LUT_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_model_index_sha256"] = "c" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        mismatch = run_preflight(
            script,
            stagepack_root,
            w8lut_root,
            rank,
            layer,
        )
        assert mismatch.returncode != 0
        assert "StagePack/W8LUT source identity mismatch" in mismatch.stderr
        manifest["source_model_index_sha256"] = source_hash
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        file_descriptor = os.open(pack_path, os.O_WRONLY)
        try:
            write_at(
                file_descriptor,
                0,
                tool.build_pack_header(layer + 1, 1024),
            )
        finally:
            os.close(file_descriptor)
        wrong_layer = run_preflight(
            script,
            stagepack_root,
            w8lut_root,
            rank,
            layer,
        )
        assert wrong_layer.returncode != 0
        assert f"W8LUT layer {layer} header mismatch" in wrong_layer.stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
