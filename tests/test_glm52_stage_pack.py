#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import tempfile


def load_stage_pack_module():
    repository = Path(__file__).resolve().parents[1]
    tool_path = repository / "runtime" / "pack" / "stage_pack.py"
    specification = importlib.util.spec_from_file_location(
        "sparkpipe_glm52_stage_pack",
        tool_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("failed to load GLM-5.2 stage-pack tool")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
) -> None:
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, body) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(body)],
        }
        payload.extend(body)
        offset += len(body)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


def write_model(model_dir: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    model_dir.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    write_safetensors(model_dir / shard_name, tensors)
    index = {
        "metadata": {
            "total_size": sum(len(item[2]) for item in tensors.values()),
        },
        "weight_map": {name: shard_name for name in tensors},
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )


def make_arguments(
    model_dir: Path,
    output_dir: Path,
    model_quantization: str,
    stages: str,
    *,
    mtp_only: bool = False,
):
    return type(
        "Arguments",
        (),
        {
            "model_dir": model_dir,
            "output_dir": output_dir,
            "model_quantization": model_quantization,
            "stages": stages,
            "reuse": False,
            "mtp_only": mtp_only,
        },
    )()


def main() -> int:
    module = load_stage_pack_module()
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        model_dir = root / "bf16-model"
        tensors = {
            "model.embed_tokens.weight": ("BF16", [4, 2], b"abcdefghijklmnop"),
            "model.layers.18.input_layernorm.weight": ("BF16", [4], b"12345678"),
            "model.layers.18.self_attn.q_proj.weight": ("BF16", [2, 2], b"QWERASDF"),
            "model.layers.18.mlp.experts.0.gate_proj.weight": ("U8", [2, 2], b"DROP"),
            "model.layers.78.enorm.weight": ("BF16", [4], b"MTPENORM"),
            "model.layers.78.self_attn.indexer.k_norm.weight": ("BF16", [4], b"MTPINDEX"),
            "model.layers.78.mlp.experts.0.gate_proj.weight": ("U8", [2, 2], b"SKIP"),
            "model.norm.weight": ("BF16", [4], b"abcdefgh"),
            "lm_head.weight": ("BF16", [8, 2], b"ABCDEFGHIJKLMNOP"),
        }
        write_model(model_dir, tensors)

        output_dir = root / "fp8-stagepacks"
        result = module.build_stage_packs(
            make_arguments(
                model_dir,
                output_dir,
                module.MODEL_QUANTIZATION_FP8,
                "0,3,12",
            )
        )
        assert result["format"] == module.FORMAT
        assert result["non_expert_weight_dtype"] == "BF16"
        index = json.loads(
            (output_dir / module.INDEX_FILE).read_text(encoding="utf-8")
        )
        tensor_map = index["tensor_map"]
        assert index["model_quantization"] == module.MODEL_QUANTIZATION_FP8
        assert index["non_expert_weight_dtype"] == "BF16"
        assert tensor_map["model.layers.18.self_attn.q_proj.weight"]["dtype"] == "BF16"
        assert "model.layers.18.mlp.experts.0.gate_proj.weight" not in tensor_map
        assert "model.layers.78.self_attn.indexer.k_norm.weight" not in tensor_map
        assert "model.layers.78.mlp.experts.0.gate_proj.weight" not in tensor_map
        assert module.MTP_EMBEDDING_ALIAS in tensor_map
        assert tensor_map["model.embed_tokens.weight"]["file"] == "stage_00_non_moe.spstage"
        assert tensor_map["model.layers.18.input_layernorm.weight"]["file"] == "stage_03_non_moe.spstage"
        assert tensor_map["lm_head.weight"]["file"] == "stage_12_non_moe.spstage"
        assert tensor_map[module.MTP_EMBEDDING_ALIAS]["file"] == "stage_12_non_moe.spstage"

        supplement_dir = root / "supplement"
        supplement_dir.mkdir()
        (supplement_dir / "stage_12_non_moe.spstage").write_bytes(b"base")
        (supplement_dir / module.INDEX_FILE).write_text(
            json.dumps(
                {
                    "format": module.FORMAT,
                    "model_quantization": module.MODEL_QUANTIZATION_FP8,
                    "topology": "ring_fixed6",
                    "stage_count": module.STAGE_COUNT,
                    "layers_per_stage": module.LAYERS_PER_STAGE,
                    "tensor_map": {},
                    "stages": {"12": {"file": "stage_12_non_moe.spstage"}},
                }
            ),
            encoding="utf-8",
        )
        module.build_stage_packs(
            make_arguments(
                model_dir,
                supplement_dir,
                module.MODEL_QUANTIZATION_FP8,
                "12",
                mtp_only=True,
            )
        )
        supplement_index = json.loads(
            (supplement_dir / module.INDEX_FILE).read_text(encoding="utf-8")
        )
        assert (supplement_dir / module.MTP_STAGE_FILE).stat().st_size > 0
        assert supplement_index["supplements"]["mtp"]["layer"] == 78
        assert supplement_index["tensor_map"]["model.layers.78.enorm.weight"]["file"] == module.MTP_STAGE_FILE
        assert supplement_index["tensor_map"][module.MTP_EMBEDDING_ALIAS]["file"] == module.MTP_STAGE_FILE

        for quantization in (
            module.MODEL_QUANTIZATION_W8,
            module.MODEL_QUANTIZATION_NVFP4,
        ):
            quantized_output = root / f"{quantization}-stagepacks"
            quantized_result = module.build_stage_packs(
                make_arguments(
                    model_dir,
                    quantized_output,
                    quantization,
                    "3",
                )
            )
            quantized_index = json.loads(
                (quantized_output / module.INDEX_FILE).read_text(encoding="utf-8")
            )
            assert quantized_result["non_expert_weight_dtype"] == "BF16"
            assert quantized_index["model_quantization"] == quantization
            assert quantized_index["non_expert_weight_dtype"] == "BF16"
            assert quantized_index["tensor_map"][
                "model.layers.18.self_attn.q_proj.weight"
            ]["dtype"] == "BF16"

        bad_model_dir = root / "non-bf16-model"
        bad_tensors = dict(tensors)
        bad_tensors["model.layers.18.self_attn.q_proj.weight"] = (
            "U8",
            [2, 2],
            b"QWER",
        )
        write_model(bad_model_dir, bad_tensors)
        for quantization in (
            module.MODEL_QUANTIZATION_FP8,
            module.MODEL_QUANTIZATION_W8,
            module.MODEL_QUANTIZATION_NVFP4,
        ):
            try:
                module.build_stage_packs(
                    make_arguments(
                        bad_model_dir,
                        root / f"bad-{quantization}",
                        quantization,
                        "3",
                    )
                )
            except module.StagePackFailure as error:
                assert "must be BF16" in str(error)
            else:
                raise AssertionError(
                    f"{quantization} stage pack accepted a non-BF16 non-expert weight"
                )
    print("PASS GLM-5.2 stage-pack precision contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
