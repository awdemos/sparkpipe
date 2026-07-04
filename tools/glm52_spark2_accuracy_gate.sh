#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${GLM52_MODEL_DIR:-/home/spark2/models/hf/nvidia/GLM-5.2-NVFP4}"
OUTPUT_DIR="${GLM52_ACCURACY_OUTPUT_DIR:-$ROOT/build/glm52_accuracy_gate}"
INPUT_HIDDEN="${GLM52_ACCURACY_INPUT_HIDDEN_BF16:-$ROOT/build/glm52_local_pipeline_gate/run1_after_layer_71.bf16}"
B12X_PACK_DIR="${GLM52_B12X_MOE_PACK_DIR:-/home/spark2/sparkpipe_artifacts/glm52_b12x_resident_moe_all_v3}"
REPEAT_COUNT="${GLM52_ACCURACY_REPEAT_COUNT:-3}"
TRACE_BUFFERS="${GLM52_ACCURACY_TRACE_BUFFERS:-1}"
MAX_STAGE_US="${GLM52_ACCURACY_MAX_STAGE_US:-1000000}"
MAX_ABS_DIFF="${GLM52_ACCURACY_MAX_ABS_DIFF:-0.01}"
MAX_MEAN_ABS_DIFF="${GLM52_ACCURACY_MAX_MEAN_ABS_DIFF:-0.001}"
MAX_LOGIT_ERROR="${GLM52_ACCURACY_MAX_LOGIT_ERROR:-0.01}"
REQUIRE_BIT_STABLE="${GLM52_ACCURACY_REQUIRE_BIT_STABLE:-1}"
NVCC_BIN="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUDA_ARCH_VALUE="${CUDA_ARCH:-sm_121a}"

MODULE="$ROOT/build/modules/glm52_resident_decode_stage/libglm52_resident_decode_stage.a"
DRIVER="$ROOT/build/packages/glm52_resident_decode_stage/stages/stage_000/model_driver.so"
VALIDATOR="$ROOT/modules/glm52_resident_decode_stage/validation/validate_glm52_resident_decode_stage_cuda.sh"
LINK_ARGS_FILE="$ROOT/build/glm52_b12x_aot/generated/runtime_link_args.txt"
B12X_ADAPTER="$ROOT/build/modules/glm52_sm121_flashinfer_b12x_moe/libglm52_sm121_flashinfer_b12x_moe_adapter.a"
B12X_BACKEND="$ROOT/build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_compiled_backend.a"
B12X_TABLE="$ROOT/build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_generated_kernel_table.a"

case "$MODEL_DIR" in
    /mnt/mac/*|/Volumes/*)
        echo "refusing remote GLM52 model dir for accuracy gate: $MODEL_DIR" >&2
        exit 2
        ;;
esac

if [ ! -d "$MODEL_DIR" ]; then
    echo "missing GLM52 model dir: $MODEL_DIR" >&2
    exit 3
fi
if [ ! -f "$INPUT_HIDDEN" ]; then
    echo "missing exact-PP13 final-stage input hidden, running local pipeline gate once: $INPUT_HIDDEN" >&2
    GLM52_MODEL_DIR="$MODEL_DIR" \
    GLM52_B12X_MOE_PACK_DIR="$B12X_PACK_DIR" \
    GLM52_ENABLE_CUDA_GRAPH_REPLAY=1 \
    GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT=1 \
    NVCC="$NVCC_BIN" \
    CUDA_ARCH="$CUDA_ARCH_VALUE" \
        bash "$ROOT/tools/glm52_spark2_local_pipeline_gate.sh" >&2
fi
if [ ! -f "$INPUT_HIDDEN" ]; then
    echo "missing exact-PP13 final-stage input hidden after local pipeline gate: $INPUT_HIDDEN" >&2
    exit 4
fi
if [ ! -d "$B12X_PACK_DIR" ]; then
    echo "missing B12x pack dir: $B12X_PACK_DIR" >&2
    exit 5
fi
for required_file in "$MODULE" "$DRIVER" "$VALIDATOR" "$LINK_ARGS_FILE" "$B12X_ADAPTER" "$B12X_BACKEND" "$B12X_TABLE"; do
    if [ ! -f "$required_file" ]; then
        echo "missing required accuracy input: $required_file" >&2
        exit 6
    fi
done

mkdir -p "$OUTPUT_DIR"
LINK_ARGS="$B12X_ADAPTER $B12X_BACKEND $B12X_TABLE $(cat "$LINK_ARGS_FILE")"

echo "accuracy_gate_model_dir=$MODEL_DIR"
echo "accuracy_gate_input_hidden=$INPUT_HIDDEN"
echo "accuracy_gate_output_dir=$OUTPUT_DIR"
echo "accuracy_gate_final_stage=72:6"
echo "accuracy_gate_graph_replay=required"
echo "accuracy_gate_max_abs_diff=$MAX_ABS_DIFF"
echo "accuracy_gate_max_mean_abs_diff=$MAX_MEAN_ABS_DIFF"
echo "accuracy_gate_max_logit_error=$MAX_LOGIT_ERROR"

rm -f "$OUTPUT_DIR"/final_repeat_*.bf16 "$OUTPUT_DIR"/final_repeat_*.log "$OUTPUT_DIR"/final_repeat.sha256 "$OUTPUT_DIR"/final_repeat.tokens
rm -f "$OUTPUT_DIR"/final_repeat.logits

for repeat_index in $(seq 1 "$REPEAT_COUNT"); do
    output_hidden="$OUTPUT_DIR/final_repeat_${repeat_index}.bf16"
    output_log="$OUTPUT_DIR/final_repeat_${repeat_index}.log"
    GLM52_REQUIRED_CUDA_LINK_ARGS="$LINK_ARGS" \
    GLM52_B12X_MOE_PACK_DIR="$B12X_PACK_DIR" \
    GLM52_MODEL_DIR="$MODEL_DIR" \
    GLM52_ROUTED_CHAIN_FIRST_LAYER_INDEX=72 \
    GLM52_ROUTED_CHAIN_LAYER_COUNT=6 \
    GLM52_ENABLE_CUDA_GRAPH_REPLAY=1 \
    GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT=1 \
    GLM52_EXACT_PP13_STAGE_SLICE=1 \
    GLM52_EXACT_PP13_STAGE_SLICE_FINAL_TOKEN=1 \
    GLM52_ACCURACY_TRACE_BUFFERS="$TRACE_BUFFERS" \
    GLM52_PIPELINE_INPUT_HIDDEN_BF16="$INPUT_HIDDEN" \
    GLM52_PIPELINE_OUTPUT_HIDDEN_BF16="$output_hidden" \
    NVCC="$NVCC_BIN" \
    CUDA_ARCH="$CUDA_ARCH_VALUE" \
	    "$VALIDATOR" "$MAX_STAGE_US" "$MODULE" "$DRIVER" \
	        >"$output_log" 2>&1
	    pass_line="$(grep -E "^glm52_resident_decode_stage validation passed.*exact_pp13_stage_slice=1.*final_stage=1" "$output_log" | tail -1 || true)"
	    if [ -z "$pass_line" ]; then
	        echo "accuracy_gate_exact_final=failed repeat=$repeat_index log=$output_log" >&2
	        tail -80 "$output_log" >&2 || true
	        exit 12
	    fi
	    printf "%s\n" "$pass_line"
	    if ! printf "%s\n" "$pass_line" | grep -E "real_lm_head=1" >/dev/null; then
	        echo "accuracy_gate_real_lm_head=failed repeat=$repeat_index" >&2
	        exit 13
	    fi
	    graph_replays="$(printf "%s\n" "$pass_line" | sed -n 's/.*graph_replays=\([0-9][0-9]*\).*/\1/p')"
	    if [ -z "$graph_replays" ] || [ "$graph_replays" = "0" ]; then
	        echo "accuracy_gate_graph_replay=failed repeat=$repeat_index graph_replays=${graph_replays:-missing}" >&2
	        exit 14
	    fi
	    logit_error="$(printf "%s\n" "$pass_line" | sed -n 's/.*real_lm_head_max_logit_error=\([^ ]*\).*/\1/p')"
	    if [ -z "$logit_error" ]; then
	        echo "accuracy_gate_real_lm_head_logit_error=missing repeat=$repeat_index" >&2
	        exit 15
	    fi
	    if ! awk -v value="$logit_error" -v limit="$MAX_LOGIT_ERROR" 'BEGIN { exit(value <= limit ? 0 : 1) }'; then
	        echo "accuracy_gate_real_lm_head_logit_error=failed repeat=$repeat_index value=$logit_error limit=$MAX_LOGIT_ERROR" >&2
	        exit 16
	    fi
	    printf "%s\n" "$logit_error" >>"$OUTPUT_DIR/final_repeat.logits"
	    sed -n 's/.*restricted_token=\([0-9][0-9]*\).*/\1/p' "$output_log" | tail -1 >>"$OUTPUT_DIR/final_repeat.tokens"
	    sha256sum "$output_hidden" >>"$OUTPUT_DIR/final_repeat.sha256"
done

unique_hash_count="$(awk '{print $1}' "$OUTPUT_DIR/final_repeat.sha256" | sort -u | wc -l | tr -d ' ')"
unique_token_count="$(sort -u "$OUTPUT_DIR/final_repeat.tokens" | wc -l | tr -d ' ')"
cat "$OUTPUT_DIR/final_repeat.sha256"
if [ "$unique_token_count" != "1" ]; then
	echo "accuracy_gate_restricted_token_stability=failed unique_token_count=$unique_token_count" >&2
	cat "$OUTPUT_DIR/final_repeat.tokens" >&2
	exit 7
fi
echo "accuracy_gate_restricted_token_stability=passed token=$(head -1 "$OUTPUT_DIR/final_repeat.tokens")"
echo "accuracy_gate_real_lm_head_logit_error=passed max=$(sort -nr "$OUTPUT_DIR/final_repeat.logits" | head -1) limit=$MAX_LOGIT_ERROR"
if [ "$unique_hash_count" != "1" ]; then
	    if [ "$TRACE_BUFFERS" != "0" ] && [ "$REPEAT_COUNT" -ge 2 ]; then
        awk '
            /accuracy_trace_buffer/ {
                layer=""; name=""; hash="";
                for (i=1; i<=NF; i++) {
                    split($i, field, "=");
                    if (field[1] == "layer") layer=field[2];
                    if (field[1] == "name") name=field[2];
                    if (field[1] == "hash64") hash=field[2];
                }
                if (layer != "" && name != "" && hash != "")
                    print layer, name, hash;
            }
        ' "$OUTPUT_DIR/final_repeat_1.log" >"$OUTPUT_DIR/final_repeat_1.trace.tsv"
        awk '
            /accuracy_trace_buffer/ {
                layer=""; name=""; hash="";
                for (i=1; i<=NF; i++) {
                    split($i, field, "=");
                    if (field[1] == "layer") layer=field[2];
                    if (field[1] == "name") name=field[2];
                    if (field[1] == "hash64") hash=field[2];
                }
                if (layer != "" && name != "" && hash != "")
                    print layer, name, hash;
            }
        ' "$OUTPUT_DIR/final_repeat_2.log" >"$OUTPUT_DIR/final_repeat_2.trace.tsv"
        awk '
            NR == FNR {
                previous[$1 " " $2] = $3;
                next;
            }
            {
                key = $1 " " $2;
                if ((key in previous) && previous[key] != $3) {
                    printf("accuracy_gate_first_divergence=layer:%s buffer:%s repeat1:%s repeat2:%s\n", $1, $2, previous[key], $3) > "/dev/stderr";
                    exit;
                }
            }
	        ' "$OUTPUT_DIR/final_repeat_1.trace.tsv" "$OUTPUT_DIR/final_repeat_2.trace.tsv"
	    fi
	    echo "accuracy_gate_routed_bit_stability=failed unique_hash_count=$unique_hash_count"
	    if [ "$REQUIRE_BIT_STABLE" != "0" ]; then
	        echo "same-input routed hidden output is not bit-stable and GLM52_ACCURACY_REQUIRE_BIT_STABLE is set" >&2
	        exit 8
	    fi
else
	echo "accuracy_gate_routed_bit_stability=passed"
fi

python3 - "$OUTPUT_DIR" "$REPEAT_COUNT" "$MAX_ABS_DIFF" "$MAX_MEAN_ABS_DIFF" <<'PY'
import math
import struct
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
repeat_count = int(sys.argv[2])
max_abs_limit = float(sys.argv[3])
max_mean_abs_limit = float(sys.argv[4])
reference = (output_dir / "final_repeat_1.bf16").read_bytes()
if len(reference) == 0 or (len(reference) % 2) != 0:
    print("accuracy_gate_routed_numeric_repeatability=failed invalid_reference", file=sys.stderr)
    raise SystemExit(9)

def bf16_to_float(value: int) -> float:
    return struct.unpack("<f", (value << 16).to_bytes(4, "little"))[0]

worst_path = ""
worst_max = 0.0
worst_mean = 0.0
for repeat_index in range(2, repeat_count + 1):
    path = output_dir / f"final_repeat_{repeat_index}.bf16"
    candidate = path.read_bytes()
    if len(candidate) != len(reference):
        print(f"accuracy_gate_routed_numeric_repeatability=failed size_mismatch repeat={repeat_index}", file=sys.stderr)
        raise SystemExit(10)
    max_abs = 0.0
    sum_abs = 0.0
    for offset in range(0, len(reference), 2):
        left = int.from_bytes(reference[offset:offset + 2], "little")
        right = int.from_bytes(candidate[offset:offset + 2], "little")
        delta = abs(bf16_to_float(left) - bf16_to_float(right))
        sum_abs += delta
        if delta > max_abs:
            max_abs = delta
    mean_abs = sum_abs / (len(reference) // 2)
    if max_abs > worst_max:
        worst_max = max_abs
        worst_mean = mean_abs
        worst_path = str(path)
if worst_max > max_abs_limit or worst_mean > max_mean_abs_limit:
    print(
        "accuracy_gate_routed_numeric_repeatability=failed "
        f"max_abs_diff={worst_max:.9g} mean_abs_diff={worst_mean:.9g} "
        f"max_abs_limit={max_abs_limit:.9g} mean_abs_limit={max_mean_abs_limit:.9g} "
        f"worst_path={worst_path}",
        file=sys.stderr,
    )
    raise SystemExit(11)
print(
    "accuracy_gate_routed_numeric_repeatability=passed "
    f"max_abs_diff={worst_max:.9g} mean_abs_diff={worst_mean:.9g} "
    f"max_abs_limit={max_abs_limit:.9g} mean_abs_limit={max_mean_abs_limit:.9g}"
)
PY

echo "accuracy_gate_routed_repeatability=passed"
