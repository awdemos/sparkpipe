#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    validator = (
        root
        / "modules"
        / "glm52_resident_decode_stage"
        / "validation"
        / "spark_glm52_resident_decode_stage_cuda_validation.cu"
    ).read_text(encoding="utf-8")
    makefile = (
        root
        / "modules"
        / "glm52_resident_decode_stage"
        / "Makefile"
    ).read_text(encoding="utf-8")
    required_cuda = (
        root
        / "modules"
        / "glm52_resident_decode_stage"
        / "source"
        / "spark_glm52_sm121_required_decode_stage.cu"
    ).read_text(encoding="utf-8")
    assert "GLM52_CHAIN_ROUTED_FROM_HIDDEN_FINAL_TOKEN" in validator
    assert "routed_pipeline_from_hidden_final=1 final_stage=1" in validator
    assert "SparkValidationSetOutputHiddenOnly" in validator
    assert "run_final_outputs = final_token_stage != 0u" in validator
    assert "SparkValidationLoadExactFullVocabFinalEpilogue" in validator
    assert "runtime->full_lm_head_weight_bf16" in validator
    assert "runtime->full_vocab_token_ids" in validator
    assert (
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT *\n"
        "        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION"
        in validator
    )
    assert "GLM52_EXACT_PP13_EXPECTED_FULL_VOCAB_TOKEN" in validator
    assert (
        "pipeline_output_hidden=%s graph_captures=0 graph_replays=0"
        not in validator
    )
    assert (
        "exact PP13 stage sequence requested graph replay but observed"
        in validator
    )
    assert "GLM52_CHAIN_ROUTED_FROM_HIDDEN_FINAL_TOKEN=1" in makefile
    assert "routed_from_hidden_final" in makefile
    assert "GLM52_CHAIN_ROUTED_FROM_HIDDEN_BF16=1 GLM52_CHAIN_ROUTED_FROM_HIDDEN_FINAL_TOKEN=1" not in makefile
    assert (
        "#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_DENSE_ROW_STRIDE 1u"
        in required_cuda
    )
    assert required_cuda.count("uint32_t candidate_row_stride") == 2
    assert required_cuda.count(
        "candidate_tokens,\n"
        "        SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_DENSE_ROW_STRIDE,\n"
    ) == 4
    assert required_cuda.count(
        "candidate_tokens,\n"
        "        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u,\n"
    ) == 1
    assert "(uint64_t)candidate_row_stride" in required_cuda
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
