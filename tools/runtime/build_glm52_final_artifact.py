#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

REQUIRED_ARCHITECTURE = "sm_121a"
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary_file:
        json.dump(value, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_name = temporary_file.name
    os.replace(temporary_name, path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one exact no-undefined GLM 5.2 sm_121a artifact."
    )
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--architecture", default=REQUIRED_ARCHITECTURE)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--link-argument", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.architecture != REQUIRED_ARCHITECTURE:
        raise SystemExit(
            f"architecture must be exactly {REQUIRED_ARCHITECTURE}, "
            f"not {arguments.architecture}"
        )
    input_paths = [path.resolve() for path in arguments.input]
    for input_path in input_paths:
        if not input_path.is_file():
            raise SystemExit(f"missing link input: {input_path}")
    arguments.output = arguments.output.resolve()
    arguments.manifest = arguments.manifest.resolve()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    version_result = subprocess.run(
        [arguments.nvcc, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    command = [
        arguments.nvcc,
        "-shared",
        f"-arch={REQUIRED_ARCHITECTURE}",
        "-Xlinker",
        "--no-undefined",
        "-o",
        str(arguments.output),
    ]
    command.extend(str(path) for path in input_paths)
    command.extend(arguments.link_argument)
    subprocess.run(command, check=True)
    if not arguments.output.is_file() or arguments.output.stat().st_size == 0:
        raise SystemExit("link command did not produce a nonempty artifact")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": "glm-5.2",
        "architecture": REQUIRED_ARCHITECTURE,
        "no_undefined": True,
        "precision": {
            "routed_expert_weights": "fp8_e4m3",
            "expert_activations": "bf16",
            "nonexpert_weights": "bf16",
            "nonexpert_activations": "bf16",
            "accumulators": "fp32",
        },
        "compiler": {
            "command": arguments.nvcc,
            "version_output": version_result.stdout + version_result.stderr,
        },
        "link_command": command,
        "inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_paths
        ],
        "artifact": {
            "path": str(arguments.output),
            "bytes": arguments.output.stat().st_size,
            "sha256": sha256_file(arguments.output),
        },
    }
    write_json_atomic(arguments.manifest, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
