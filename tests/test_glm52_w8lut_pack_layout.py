#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


def load_tool(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    tool_path = repo_root / "tools" / f"{name}.py"
    sys.path.insert(0, str(tool_path.parent))
    spec = importlib.util.spec_from_file_location(name, tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeReader:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir

    def tensor(self, name: str):
        return name

    def close(self) -> None:
        return


def main() -> int:
    packer = load_tool("glm52_w8lut_resident_pack")
    fp8 = load_tool("glm52_fp8_resident_pack")
    codec = load_tool("glm52_w8lut_codec")
    assert packer.WIRE_MAGIC != fp8.WIRE_MAGIC
    assert packer.PACK_FILE_TEMPLATE.endswith(".spw8lut")
    assert not packer.PACK_FILE_TEMPLATE.endswith(".spfp8")
    assert packer.MANIFEST_FILE != "fp8_moe_pack_manifest.json"
    assert packer.QUANT_MODE_W8LUT != fp8.QUANT_MODE_FP8_E4M3
    assert packer.parse_layers("3,4;5") == [3, 4, 5]
    try:
        packer.parse_layers("3,3")
    except packer.PackFailure:
        pass
    else:
        raise AssertionError("duplicate W8LUT layer selection was accepted")
    values = np.array([0x0000, 0x8000, 0x3F80, 0xBF80, 0x4000, 0x4040], dtype=np.uint16)
    codes, e0, _ = codec.encode(values)
    codec.verify(values, codes, e0)
    decoded = codec.decode(codes, e0)
    recoded, re0, _ = codec.encode(decoded)
    assert int(re0) == int(e0)
    assert np.array_equal(recoded, codes)
    regions = packer.reserve_regions()
    header = packer.pack_header(17, regions, 1024)
    fields, decoded_regions = packer.unpack_pack_header(header)
    assert fields.magic == packer.WIRE_MAGIC
    assert fields.layer_index == 17
    assert fields.maximum_token_count == 1024
    assert fields.quant_mode == packer.QUANT_MODE_W8LUT
    assert fields.scale_layout == packer.SCALE_LAYOUT_W8LUT_EXPERT_COMPONENT_E0
    assert decoded_regions == regions
    assert all(region["offset"] % packer.REGION_ALIGNMENT == 0 for region in regions)
    with tempfile.TemporaryDirectory(prefix="sparkpipe_w8lut_pack_test_") as temp_dir:
        root = Path(temp_dir)
        mixed = root / "mixed"
        mixed.mkdir()
        (mixed / "glm52_layer_0003_fp8_moe.spfp8").touch()
        try:
            packer.validate_output_directory(mixed)
        except packer.PackFailure:
            pass
        else:
            raise AssertionError("W8LUT packer accepted an FP8 pack directory")
        dedicated = root / "dedicated"
        dedicated.mkdir()
        (dedicated / ".glm52_layer_0003_w8lut_worker.tmp").touch()
        packer.validate_output_directory(dedicated)
        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {}}),
            encoding="utf-8",
        )
        output_dir = root / "w8lut"
        packer.EXPERT_COUNT = 2
        packer.HIDDEN_DIMENSION = 4
        packer.INTERMEDIATE_DIMENSION = 3
        packer.REGION_ALIGNMENT = 16
        packer.SafetensorReader = FakeReader

        def fake_encode_tensor(tensor, expected_shape, name):
            count = int(np.prod(expected_shape))
            codes = np.arange(count, dtype=np.uint8)
            return codes, 120, {
                "element_count": count,
                "below_window_count": 0,
            }

        packer.encode_tensor = fake_encode_tensor
        record = packer.write_layer_pack(model_dir, output_dir, 3, 1024)
        pack_path = output_dir / "glm52_layer_0003_w8lut_moe.spw8lut"
        assert pack_path.is_file()
        assert record["path"] == str(pack_path)
        assert record["bytes"] == pack_path.stat().st_size
        with pack_path.open("rb") as file:
            fields, file_regions = packer.unpack_pack_header(file.read(packer.HEADER_BYTES))
        assert fields.magic == packer.WIRE_MAGIC
        assert fields.layer_index == 3
        assert file_regions == packer.reserve_regions()
        try:
            packer.write_layer_pack(model_dir, output_dir, 3, 1024)
        except packer.PackFailure:
            pass
        else:
            raise AssertionError("W8LUT packer overwrote an existing pack")
        packer.write_manifest(output_dir, model_dir, 1024, [record])
        manifest = json.loads((output_dir / packer.MANIFEST_FILE).read_text(encoding="utf-8"))
        assert manifest["pack_extension"] == ".spw8lut"
        assert manifest["source_dtype"] == "BF16"
        assert manifest["layers"][0]["file"] == pack_path.name
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
