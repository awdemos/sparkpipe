#!/usr/bin/env python3

from pathlib import Path


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    makefile = (repository / "Makefile").read_text(encoding="utf-8")
    validation = (
        repository /
        "modules/glm52_resident_decode_stage/validation/"
        "spark_glm52_resident_decode_stage_cuda_validation.cu"
    ).read_text(encoding="utf-8")
    assert "GLM52_W8LUT_MOE_PACK_DIR ?=" in makefile
    assert makefile.count(
        "GLM52_W8LUT_MOE_PACK_DIR='$(GLM52_W8LUT_MOE_PACK_DIR)'"
    ) == 3
    assert "routed_fixture_layer_index =" in validation
    assert "? routed_chain_first_layer_index" in validation
    assert "SparkValidationLoadRoutedLayerRouterBf16Fixture(\n" in validation
    assert "routed_fixture_layer_index,\n            &layer3_router" in validation
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
