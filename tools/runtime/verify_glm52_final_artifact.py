#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_ARCHITECTURE = "sm_121a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_file_record(record: object, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{label} is not an object")
    path_value = record.get("path")
    bytes_value = record.get("bytes")
    sha256_value = record.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label}.path is invalid")
    if not isinstance(bytes_value, int) or bytes_value <= 0:
        raise ValueError(f"{label}.bytes is invalid")
    if not isinstance(sha256_value, str) or len(sha256_value) != 64:
        raise ValueError(f"{label}.sha256 is invalid")
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    if path.stat().st_size != bytes_value:
        raise ValueError(f"{label} byte count changed: {path}")
    if sha256_file(path) != sha256_value:
        raise ValueError(f"{label} SHA-256 changed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    with arguments.manifest.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("schema_version") != 1:
        raise SystemExit("unsupported manifest schema")
    if manifest.get("model") != "glm-5.2":
        raise SystemExit("manifest is not for GLM 5.2")
    if manifest.get("architecture") != REQUIRED_ARCHITECTURE:
        raise SystemExit("manifest architecture is not sm_121a")
    if manifest.get("no_undefined") is not True:
        raise SystemExit("manifest does not require no-undefined linking")
    precision = manifest.get("precision")
    expected_precision = {
        "routed_expert_weights": "fp8_e4m3",
        "expert_activations": "bf16",
        "nonexpert_weights": "bf16",
        "nonexpert_activations": "bf16",
        "accumulators": "fp32",
    }
    if precision != expected_precision:
        raise SystemExit("manifest precision policy is not FP8-expert/BF16-rest")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise SystemExit("manifest has no link inputs")
    for input_index, input_record in enumerate(inputs):
        validate_file_record(input_record, f"inputs[{input_index}]")
    validate_file_record(manifest.get("artifact"), "artifact")
    print("PASS GLM 5.2 final artifact identity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
