#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile


def load_tool(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "tools" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    watcher = load_tool("glm52_w8lut_stage_pack_watch")
    assert watcher.expected_expert_layers(0) == [3, 4, 5]
    assert watcher.expected_expert_layers(1) == [6, 7, 8, 9, 10, 11]
    assert watcher.expected_expert_layers(12) == [72, 73, 74, 75, 76, 77, 78]
    with tempfile.TemporaryDirectory(prefix="sparkpipe_w8lut_watch_test_") as tmp:
        model_dir = Path(tmp)
        weight_map = {
            "model.layers.12.input_layernorm.weight": "stage2_nonexpert.safetensors",
            "model.layers.70.input_layernorm.weight": "unrelated.safetensors",
            "model.norm.weight": "final.safetensors",
        }
        for layer in watcher.expected_expert_layers(2):
            for expert in range(watcher.EXPERT_COUNT):
                for projection in watcher.EXPERT_PROJECTIONS:
                    name = watcher.tensor_name(
                        layer, expert, projection, "weight"
                    )
                    weight_map[name] = f"layer{layer}.safetensors"
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}),
            encoding="utf-8",
        )
        required = watcher.exact_pack_required_files(
            model_dir,
            2,
            watcher.expected_expert_layers(2),
            True,
        )
        assert required == [
            "layer12.safetensors",
            "layer13.safetensors",
            "layer14.safetensors",
            "layer15.safetensors",
            "layer16.safetensors",
            "layer17.safetensors",
            "model.safetensors.index.json",
            "stage2_nonexpert.safetensors",
        ]
        assert "unrelated.safetensors" not in required
        assert "final.safetensors" not in required
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
