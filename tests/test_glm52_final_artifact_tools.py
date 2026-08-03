#!/usr/bin/env python3

import json
from pathlib import Path
import subprocess
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
BUILD_TOOL = REPOSITORY / "tools/runtime/build_glm52_final_artifact.py"
VERIFY_TOOL = REPOSITORY / "tools/runtime/verify_glm52_final_artifact.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary_directory_text:
        temporary_directory = Path(temporary_directory_text)
        fake_nvcc = temporary_directory / "nvcc"
        fake_nvcc.write_text(
            """#!/bin/sh
if [ \"$1\" = \"--version\" ]; then
    echo 'fake nvcc CUDA 13.3'
    exit 0
fi
output=''
previous=''
for argument in \"$@\"; do
    if [ \"$previous\" = '-o' ]; then
        output=\"$argument\"
    fi
    previous=\"$argument\"
done
[ -n \"$output\" ] || exit 7
printf 'ELF-sm_121a-no-undefined' > \"$output\"
""",
            encoding="utf-8",
        )
        fake_nvcc.chmod(0o755)
        first_input = temporary_directory / "first.o"
        second_input = temporary_directory / "second.a"
        first_input.write_bytes(b"first")
        second_input.write_bytes(b"second")
        output = temporary_directory / "libglm52.so"
        manifest = temporary_directory / "manifest.json"
        subprocess.run(
            [
                "python3",
                str(BUILD_TOOL),
                "--nvcc",
                str(fake_nvcc),
                "--architecture",
                "sm_121a",
                "--output",
                str(output),
                "--manifest",
                str(manifest),
                "--input",
                str(first_input),
                "--input",
                str(second_input),
            ],
            check=True,
        )
        subprocess.run(
            ["python3", str(VERIFY_TOOL), str(manifest)],
            check=True,
        )
        value = json.loads(manifest.read_text(encoding="utf-8"))
        command = value["link_command"]
        assert "-arch=sm_121a" in command
        assert "--no-undefined" in command
        second_input.write_bytes(b"changed")
        failed = subprocess.run(
            ["python3", str(VERIFY_TOOL), str(manifest)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0
    print("PASS GLM 5.2 final artifact build and verification tools")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
