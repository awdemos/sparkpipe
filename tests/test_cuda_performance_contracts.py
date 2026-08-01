#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_ROOTS = (
    ROOT / "inference",
    ROOT / "runtime",
    ROOT / "modules",
)


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label} is missing {needle!r}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"{label} contains forbidden {needle!r}")


def strip_comments_and_literals(text: str) -> str:
    output: list[str] = []
    index = 0
    state = "code"
    quote = ""

    while index < len(text):
        current = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if current == "/" and following == "/":
                output.extend("  ")
                index += 2
                state = "line_comment"
                continue
            if current == "/" and following == "*":
                output.extend("  ")
                index += 2
                state = "block_comment"
                continue
            if current in ("'", '"'):
                quote = current
                output.append(" ")
                index += 1
                state = "literal"
                continue
            output.append(current)
            index += 1
            continue
        if state == "line_comment":
            if current == "\n":
                output.append("\n")
                state = "code"
            else:
                output.append(" ")
            index += 1
            continue
        if state == "block_comment":
            if current == "*" and following == "/":
                output.extend("  ")
                index += 2
                state = "code"
            else:
                output.append("\n" if current == "\n" else " ")
                index += 1
            continue
        if current == "\\" and index + 1 < len(text):
            output.extend("  ")
            index += 2
            continue
        if current == quote:
            output.append(" ")
            index += 1
            state = "code"
            continue
        output.append("\n" if current == "\n" else " ")
        index += 1

    if state == "block_comment":
        raise AssertionError("unterminated block comment")
    if state == "literal":
        raise AssertionError("unterminated source literal")
    return "".join(output)


def validate_balanced_delimiters(path: Path) -> None:
    source = strip_comments_and_literals(path.read_text(encoding="utf-8"))
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[tuple[str, int]] = []

    for offset, character in enumerate(source):
        if character in opening:
            stack.append((character, offset))
        elif character in closing:
            if not stack or stack[-1][0] != closing[character]:
                raise AssertionError(
                    f"{path.relative_to(ROOT)} has unmatched {character!r} "
                    f"at byte {offset}"
                )
            stack.pop()
    if stack:
        character, offset = stack[-1]
        raise AssertionError(
            f"{path.relative_to(ROOT)} has unmatched {character!r} "
            f"at byte {offset}"
        )


def owned_cuda_files() -> list[Path]:
    files: list[Path] = []
    for root in CUDA_ROOTS:
        if not root.exists():
            continue
        files.extend(root.rglob("*.cu"))
        files.extend(root.rglob("*.cuh"))
    return sorted(set(files))


def validate_scale_abi() -> None:
    scale = read("inference/kernels/scale.cuh")
    gemm = read("inference/kernels/gemm.cuh")
    runtime = read("runtime/gemm.cuh")

    for encoding in (
        "LM_SCALE_ENCODING_NONE",
        "LM_SCALE_ENCODING_F32",
        "LM_SCALE_ENCODING_UE4M3",
        "LM_SCALE_ENCODING_UE8M0",
    ):
        require(scale, encoding, "shared scale ABI")
    for field in (
        "group_stride_entries",
        "row_group_stride_entries",
        "group_count",
        "row_count",
        "input_dimension",
        "row_group_size",
        "k_group_size",
    ):
        require(scale, field, "shared scale tensor")
    require(scale, "(uint64_t)group_index * scale->group_stride_entries", "expert scale indexing")
    require(scale, "row_index / scale->row_group_size", "row-group scale indexing")
    require(scale, "k_index / scale->k_group_size", "K-group scale indexing")
    require(gemm, "LmScaleTensor scale_a;", "GEMM activation scale descriptor")
    require(gemm, "LmScaleTensor scale_b;", "GEMM weight scale descriptor")
    require(runtime, "LmGemmValidateScaleTensor<FormatA>", "activation scale validation")
    require(runtime, "LmGemmValidateScaleTensor<FormatB>", "weight scale validation")

    cast_pattern = re.compile(r"scale_[ab]\s*=\s*\(const\s+float\s*\*\)")
    for path in owned_cuda_files():
        text = path.read_text(encoding="utf-8")
        match = cast_pattern.search(text)
        if match:
            raise AssertionError(
                f"{path.relative_to(ROOT)} reinterprets a scale plane as float*: "
                f"{match.group(0)!r}"
            )


def validate_tma_contract() -> None:
    tma = read("inference/kernels/tma.cuh")
    tile = read("inference/kernels/tile.cuh")
    gemm = read("inference/kernels/gemm.cuh")
    runtime = read("runtime/gemm.cuh")

    tma_code = strip_comments_and_literals(tma)
    forbid(tma_code, "elect.sync", "TMA producer election")
    require(tile, "threadIdx.x == 0u", "CTA-wide producer selection")
    require(gemm, "LmPipelineInitialise<STAGES>(barrier, 1u)", "single-producer mbarrier")
    require(gemm, "__grid_constant__ const CUtensorMap tensor_map_a", "by-value activation tensor map")
    require(gemm, "__grid_constant__ const CUtensorMap tensor_map_b", "by-value weight tensor map")
    require(runtime, "alignas(64) CUtensorMap activation_map;", "activation tensor-map lifetime")
    require(runtime, "alignas(64) CUtensorMap weight_map;", "weight tensor-map lifetime")
    forbid(gemm, "const CUtensorMap *tensor_map", "device pointer to host tensor map")


def validate_quantizer_writes() -> None:
    mma = read("inference/kernels/mma.cuh")
    norm = read("inference/kernels/norm.cuh")

    require(mma, "void LmStoreCodeOctet(", "exclusive packed-code writer")
    require(mma, "const float values[8]", "eight-code ownership")
    require(mma, "memcpy(base + byte, &packed, Format::kStoredBits);", "byte-disjoint packed store")
    require(norm, "index = threadIdx.x * 8u", "eight-code quantizer scheduling")
    forbid(strip_comments_and_literals(norm), "LmStoreCodePair", "overlapping packed pair writer")

    for width in (4, 6, 7, 8):
        mask = (1 << width) - 1
        codes = [((index * 37) + width) & mask for index in range(40)]
        packed = bytearray((len(codes) * width + 7) // 8)
        for begin in range(0, len(codes), 8):
            word = 0
            for lane, code in enumerate(codes[begin:begin + 8]):
                word |= code << (lane * width)
            byte = (begin * width) // 8
            packed[byte:byte + width] = word.to_bytes(8, "little")[:width]
        decoded = []
        bits = int.from_bytes(packed, "little")
        for index in range(len(codes)):
            decoded.append((bits >> (index * width)) & mask)
        if decoded != codes:
            raise AssertionError(f"{width}-bit octet packing did not round-trip")


def validate_model_precision_contracts() -> None:
    glm = read("inference/llms/glm5_2/layer.cuh")
    k3 = read("inference/llms/kimi_k3/layer.cuh")
    qwen_bind = read("inference/llms/qwen_3_6/bind.cu")
    dsv4 = read("inference/llms/deepseek_v4/layer.cuh")
    dsv4_unity = read("inference/llms/deepseek_v4/unity.cu")
    dsv4_pro = read("inference/llms/deepseek_v4_pro/unity.cu")

    require(glm, "LmGemmWeightOnlyLaunch<\n        LmFp8,", "GLM FP8 expert weights")
    require(glm, "LmGemmLaunch<\n        LmBf16Format,", "GLM BF16 non-expert execution")
    forbid(glm, "LmQuantiseRowsKernel", "GLM BF16 activation path")

    require(k3, "LmGemmWeightOnlyLaunch<", "K3 BF16-activation/MXFP4-weight experts")
    require(k3, "LmScaleTensorBlockUe8m0(", "K3 MXFP4 scale plane")
    require(qwen_bind, "Qwen36LaunchSlice<LmBf16Format>", "Qwen 3.6 BF16 entry point")

    require(dsv4, "Dsv4Fp8ActivationScale(", "DSV4 dynamic FP8 activation scale")
    require(dsv4, "Dsv4Fp8WeightScale(", "DSV4 non-expert FP8 weight scale")
    require(dsv4, "Dsv4CheckpointFp4WeightScale(", "DSV4 checkpoint FP4 expert scale")
    require(dsv4, "LmGemmLaunchAsymmetric<", "DSV4 FP8-activation/FP4-weight expert GEMM")
    require(dsv4_unity, "Dsv4LayerMoeCheckpointFp4", "DSV4 Flash mixed-precision export")
    require(dsv4_unity, "LmGemmKernel<LmFp8, LmMxfp4", "DSV4 Flash mixed GEMM instantiation")
    require(dsv4_pro, "LmGemmKernel<\n    LmFp8,\n    LmMxfp4,", "DSV4 Pro mixed GEMM instantiation")
    forbid(dsv4, "(const float *)b->packed_scale", "DSV4 stale activation-scale cast")
    forbid(dsv4, "(const float *)b->expert_w", "DSV4 stale expert-scale cast")



def validate_k3_exact_replay() -> None:
    layer = read("inference/llms/kimi_k3/layer.cuh")
    slice_source = read("inference/llms/kimi_k3/slice.cuh")
    combined_code = strip_comments_and_literals(layer + "\n" + slice_source)

    require(layer, "float *replay_retention;", "K3 retained decay values")
    require(layer, "float *replay_write_gate;", "K3 retained write gates")
    require(layer, "? b->replay_retention : b->kda_retention", "K3 direct retention capture")
    require(layer, "? b->replay_write_gate : b->kda_write_gate_out", "K3 direct gate capture")
    require(slice_source, "buffers->replay_retention,buffers->replay_write_gate", "K3 exact fold inputs")
    require(slice_source, "K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(float)", "K3 fold shared-state allocation")
    fold_begin = slice_source.index("static int32_t K3FoldAccepted")
    fold_source = strip_comments_and_literals(slice_source[fold_begin:])
    forbid(fold_source, "LmBoundedDecayKernel", "K3 accepted-prefix gate recomputation")
    forbid(fold_source, "LmSigmoidRowsKernel", "K3 accepted-prefix beta recomputation")
    forbid(combined_code, "replay_decay_logit", "raw replay decay logits")
    forbid(combined_code, "replay_beta_logit", "raw replay beta logits")

def validate_grouped_moe_contract() -> None:
    route = read("inference/kernels/route.cuh")
    queue_header = read(
        "model-families/glm52/include/sparkpipe/spark_glm52_expert_queue.h"
    )
    queue_source = read("model-families/glm52/src/spark_glm52_expert_queue.c")

    require(route, "packed_rows != expected_packed_rows", "route cardinality validation")
    require(route, "LmLaunchGroupedTileM(rows,top_k,EXPERTS)", "token-priced grouped tile")
    require(queue_header, "SPARK_GLM52_EXPERT_QUEUE_MODE_SEALED_BATCH", "sealed expert batch mode")
    require(queue_header, "SparkGlm52ExpertQueueSealLayer", "sealed layer API")
    require(queue_source, "queue->layer_sealed[layer_index]", "sealed layer state")


def validate_stream_ordered_dispatch() -> None:
    dispatch = read("inference/stage/dispatch.cu")

    require(dispatch, "cudaLaunchHostFunc(", "stream-ordered stage completion")
    require(dispatch, "GLM52_STAGE_SLICE_DEBUG_SYNC", "debug-only synchronization gate")
    require(dispatch, "SparkResidentDecodeStageBackendQuiesce", "explicit quiesce boundary")


def main() -> int:
    files = owned_cuda_files()
    if not files:
        raise AssertionError("no owned CUDA translation units were found")
    for path in files:
        validate_balanced_delimiters(path)
    validate_scale_abi()
    validate_tma_contract()
    validate_quantizer_writes()
    validate_k3_exact_replay()
    validate_model_precision_contracts()
    validate_grouped_moe_contract()
    validate_stream_ordered_dispatch()
    print(
        f"PASS CUDA performance source contracts: {len(files)} owned CUDA files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
