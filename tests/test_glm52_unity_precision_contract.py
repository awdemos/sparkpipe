#!/usr/bin/env python3
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    unity = (ROOT / "inference/llms/glm5_2/unity.cu").read_text()
    api = (ROOT / "inference/llms/glm5_2/api.h").read_text()
    failures: list[str] = []

    required_entries = (
        "Glm52GemmBf16",
        "Glm52GemmFp8ExpertWeightBf16Activation",
        "Glm52LayerAttentionBf16",
        "Glm52LayerDenseMlpBf16",
        "Glm52LayerMoeFp8ExpertWeightBf16Activation",
    )
    for name in required_entries:
        if name not in unity:
            failures.append(f"missing precision-explicit GLM entry point {name}")

    forbidden_tokens = (
        "Glm52LayerAttentionFp8",
        "Glm52LayerDenseMlpFp8",
        "Glm52LayerMoeInt7",
        "Glm52GemmInt7",
        "Glm52GemmInt6",
        "Glm52GemmInt8",
        "Glm52GemmNvfp4",
        "LmQuantiseRowsKernel",
    )
    for token in forbidden_tokens:
        if token in unity:
            failures.append(f"GLM unity exposes forbidden precision path {token}")

    if not re.search(
        r"LmGemmWeightOnlyLaunch<\s*LmFp8\s*,",
        unity,
        re.S,
    ):
        failures.append("GLM expert API does not use the weight-only FP8 launch")
    if "if (!grouped)" not in unity:
        failures.append("GLM expert API does not reject a dense/non-grouped call")

    for name in ("Glm52GemmBf16", "Glm52GemmFp8ExpertWeightBf16Activation"):
        if name not in api:
            failures.append(f"GLM public API does not declare {name}")
    for token in ("Glm52GemmFp8(", "Glm52GemmInt7(", "Glm52GemmNvfp4("):
        if token in api:
            failures.append(f"GLM public API still declares ambiguous path {token[:-1]}")

    if failures:
        print("\n".join(failures))
        return 1
    print("PASS GLM unity exposes only BF16-rest/FP8-expert execution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
