#!/usr/bin/env python3
"""
Build a setup-time SparkPipe manifest for the GLM-5.2 DSpark speculator.

The manifest is a small JSON artifact. It is safe for production C/CUDA to
consume; Python is only used here during one-time packaging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


FORMAT = "sparkpipe.glm52.dspark.speculator_manifest.v1"
MODEL_ID = "RedHatAI/GLM-5.2-speculator.dspark"
BASE_MODEL = "zai-org/GLM-5.2-FP8"
AUX_LAYERS = [8, 23, 39, 55, 70]
MAX_SPECULATIVE_TOKENS = 7


class ManifestFailure(RuntimeError):
    pass


def require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ManifestFailure(f"{name}={actual!r}, expected {expected!r}")


def require_layer_ids(actual: Iterable[Any]) -> List[int]:
    layers = [int(value) for value in actual]
    require_equal("aux_hidden_state_layer_ids", layers, AUX_LAYERS)
    return layers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_config(model_dir: Path) -> Dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise ManifestFailure(f"missing DSpark config: {config_path}")
    return json.loads(config_path.read_text())


def build_manifest(model_dir: Path, skip_sha256: bool) -> Dict[str, Any]:
    config = load_config(model_dir)
    transformer = config.get("transformer_layer_config") or {}
    speculators = config.get("speculators_config") or {}
    proposals = speculators.get("proposal_methods") or []
    proposal = proposals[0] if proposals else {}
    verifier = speculators.get("verifier") or {}
    model_path = model_dir / "model.safetensors"

    require_equal("architectures", config.get("architectures"), ["DSparkDraftModel"])
    aux_layers = require_layer_ids(config.get("aux_hidden_state_layer_ids") or [])
    require_equal("block_size", config.get("block_size"), 8)
    require_equal("dtype", config.get("dtype"), "bfloat16")
    require_equal("draft_vocab_size", config.get("draft_vocab_size"), 154880)
    require_equal("markov_rank", config.get("markov_rank"), 256)
    require_equal("max_anchors", config.get("max_anchors"), 1024)
    require_equal("enable_confidence_head", config.get("enable_confidence_head"), True)
    require_equal(
        "confidence_head_with_markov",
        config.get("confidence_head_with_markov"),
        True,
    )
    require_equal("speculators_config.algorithm", speculators.get("algorithm"), "dspark")
    require_equal("proposal_type", proposal.get("proposal_type"), "greedy")
    require_equal(
        "proposal speculative_tokens",
        proposal.get("speculative_tokens"),
        MAX_SPECULATIVE_TOKENS,
    )
    require_equal("proposal verifier_accept_k", proposal.get("verifier_accept_k"), 1)
    require_equal("verifier.name_or_path", verifier.get("name_or_path"), BASE_MODEL)
    require_equal("draft hidden_size", transformer.get("hidden_size"), 6144)
    require_equal("draft intermediate_size", transformer.get("intermediate_size"), 12288)
    require_equal("draft num_hidden_layers", transformer.get("num_hidden_layers"), 5)
    require_equal("draft num_attention_heads", transformer.get("num_attention_heads"), 64)
    require_equal("draft num_key_value_heads", transformer.get("num_key_value_heads"), 64)
    require_equal("draft head_dim", transformer.get("head_dim"), 64)
    require_equal("draft vocab_size", transformer.get("vocab_size"), 154880)

    if not model_path.exists():
        raise ManifestFailure(f"missing DSpark weights: {model_path}")

    manifest = {
        "format": FORMAT,
        "model_id": MODEL_ID,
        "base_model": BASE_MODEL,
        "verifier_quantization": "fp8_e4m3_8bit",
        "draft_dtype": "bf16",
        "draft_architecture": "qwen3",
        "aux_hidden_state_layer_ids": aux_layers,
        "maximum_speculative_token_count": MAX_SPECULATIVE_TOKENS,
        "verifier_accept_k": 1,
        "model_safetensors": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
        },
        "contract": {
            "abi_version": 1,
            "verifier_quantization_mode": 2,
            "draft_dtype": 1,
            "draft_layer_count": 5,
            "block_size": 8,
            "hidden_dimension": 6144,
            "intermediate_dimension": 12288,
            "attention_head_count": 64,
            "kv_head_count": 64,
            "head_dimension": 64,
            "vocab_size": 154880,
            "draft_vocab_size": 154880,
            "markov_rank": 256,
            "max_anchors": 1024,
            "maximum_speculative_token_count": MAX_SPECULATIVE_TOKENS,
            "verifier_accept_k": 1,
            "enable_confidence_head": 1,
            "confidence_head_with_markov": 1,
        },
    }
    if not skip_sha256:
        manifest["model_safetensors"]["sha256"] = sha256_file(model_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a SparkPipe GLM-5.2 DSpark speculator manifest"
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(Path(args.model_dir), args.skip_sha256)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"dspark_manifest path={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
