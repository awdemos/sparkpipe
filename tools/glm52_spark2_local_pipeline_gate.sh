#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${GLM52_MODEL_DIR:-/home/spark2/models/hf/nvidia/GLM-5.2-NVFP4}"
OUTPUT_DIR="${GLM52_LOCAL_PIPELINE_OUTPUT_DIR:-$ROOT/build/glm52_local_pipeline_gate}"
INPUT_TOKEN_ID="${GLM52_LOCAL_PIPELINE_INPUT_TOKEN_ID:-1037}"
MAX_STAGE_US="${GLM52_LOCAL_PIPELINE_MAX_STAGE_US:-1000000}"
NVCC_BIN="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUDA_ARCH_VALUE="${CUDA_ARCH:-sm_121a}"
ENABLE_GRAPH_REPLAY="${GLM52_ENABLE_CUDA_GRAPH_REPLAY:-1}"
AOT_OUTPUT_DIR="${B12X_AOT_OUTPUT_DIR:-build/glm52_b12x_aot_prompt}"
B12X_PACK_DIR="/home/spark2/sparkpipe_artifacts/glm52_b12x_resident_moe_all_v3"
RESUME="${GLM52_LOCAL_PIPELINE_RESUME:-0}"
ACTIVE_SEQUENCE_COUNT="${GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT:-1}"
DIRECT_RUNNER_DIR="${GLM52_LOCAL_PIPELINE_RUNNER_DIR:-$OUTPUT_DIR/runner}"
DIRECT_RUNNER="$DIRECT_RUNNER_DIR/glm52_resident_decode_stage_runner"
MODULE_ARCHIVE="${GLM52_LOCAL_PIPELINE_MODULE_ARCHIVE:-$ROOT/build/modules/glm52_resident_decode_stage/libglm52_resident_decode_stage.a}"
FORCE_RUNNER_REBUILD="${GLM52_LOCAL_PIPELINE_FORCE_RUNNER_REBUILD:-0}"
PREFILL_TOKEN_IDS_FILE="${GLM52_PREFILL_TOKEN_IDS_FILE:-}"
PREFILL_MAX_TOKENS="${GLM52_LOCAL_PIPELINE_MAX_PREFILL_TOKENS:-256}"
PREFILL_SCHEDULE_TSV="$OUTPUT_DIR/prefill_schedule.tsv"
PREFILL_ONLY="${GLM52_LOCAL_PIPELINE_PREFILL_ONLY:-0}"
PROMPT_SEQUENCE="${GLM52_LOCAL_PIPELINE_PROMPT_SEQUENCE:-0}"

case "$MODEL_DIR" in
	/mnt/mac/*|/Volumes/*)
		echo "refusing remote GLM52 model dir for local pipeline gate: $MODEL_DIR" >&2
		exit 2
		;;
esac

if [ ! -d "$MODEL_DIR" ]; then
	echo "missing GLM52 model dir: $MODEL_DIR" >&2
	exit 3
fi
if ! command -v "$NVCC_BIN" >/dev/null 2>&1; then
	echo "missing nvcc for GLM52 local pipeline gate: $NVCC_BIN" >&2
	exit 4
fi
mkdir -p "$OUTPUT_DIR"
SUMMARY_TSV="$OUTPUT_DIR/local_pipeline_summary.tsv"
printf "stage\tmode\tfirst_layer\tlayer_count\tpack_dir\ttotal_us\tmaximum_us\ttoken\tprompt_prefill\tprompt_source_tokens\tlog\n" >"$SUMMARY_TSV"

pack_dir_has_stage_packs()
{
	local pack_dir="$1"
	local first="$2"
	local count="$3"
	local layer
	local pack_path
	if [ ! -s "$pack_dir/resident_moe_pack_manifest.json" ]; then
		return 1
	fi
	for ((layer=first; layer<first+count; layer++)); do
		if [ "$layer" -lt 3 ]; then
			continue
		fi
		pack_path="$(printf "%s/glm52_layer_%04u_b12x_moe.spb12x" "$pack_dir" "$layer")"
		if [ ! -s "$pack_path" ]; then
			return 1
		fi
	done
	return 0
}

require_stage_packs()
{
	local pack_dir="$1"
	local first="$2"
	local count="$3"
	local layer
	local pack_path
	if [ ! -s "$pack_dir/resident_moe_pack_manifest.json" ]; then
		echo "missing B12x pack manifest: $pack_dir/resident_moe_pack_manifest.json" >&2
		exit 6
	fi
	for ((layer=first; layer<first+count; layer++)); do
		if [ "$layer" -lt 3 ]; then
			continue
		fi
		pack_path="$(printf "%s/glm52_layer_%04u_b12x_moe.spb12x" "$pack_dir" "$layer")"
		if [ ! -s "$pack_path" ]; then
			echo "missing B12x resident pack: $pack_path" >&2
			exit 7
		fi
	done
}

extract_field()
{
	local name="$1"
	local text="$2"
	printf "%s" "$text" | sed -n "s/.*${name}=\([^ ]*\).*/\1/p" | tail -1
}

stage_pass_line()
{
	local log_path="$1"
	local pass_line
	pass_line="$(grep -E "^glm52_resident_decode_stage validation passed" "$log_path" | head -1 || true)"
	if [ -z "$pass_line" ]; then
		pass_line="$(grep -E "validation passed|orchestrator validation passed" "$log_path" | tail -1 || true)"
	fi
	printf "%s" "$pass_line"
}

record_stage_summary()
{
	local mode="$1"
	local first="$2"
	local count="$3"
	local pack_dir="$4"
	local log_path="$5"
	local pass_line
	local total_us
	local maximum_us
	local token
	local prompt_prefill
	local prompt_source_tokens
	pass_line="$(stage_pass_line "$log_path")"
	if [ -z "$pass_line" ]; then
		return 1
	fi
	total_us="$(extract_field "total_us" "$pass_line")"
	maximum_us="$(extract_field "maximum_us" "$pass_line")"
	token="$(extract_field "restricted_token" "$pass_line")"
	prompt_prefill="$(extract_field "prompt_prefill" "$pass_line")"
	prompt_source_tokens="$(extract_field "prompt_source_tokens" "$pass_line")"
	printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
		"${first}:${count}" "$mode" "$first" "$count" "$pack_dir" \
		"${total_us:-}" "${maximum_us:-}" "${token:-}" \
		"${prompt_prefill:-}" "${prompt_source_tokens:-}" "$log_path" >>"$SUMMARY_TSV"
	return 0
}

run_prefill_schedule()
{
	local final_kind
	local prefill_step_count
	local prefill_token_count
	if [ -z "$PREFILL_TOKEN_IDS_FILE" ]; then
		return 0
	fi
	if [ ! -s "$PREFILL_TOKEN_IDS_FILE" ]; then
		echo "missing GLM52 prefill token ids file: $PREFILL_TOKEN_IDS_FILE" >&2
		exit 18
	fi
	make -C "$ROOT" build/sparkpipe_glm52_prefill_dryrun
	if ! "$ROOT/build/sparkpipe_glm52_prefill_dryrun" \
		--tokens "$PREFILL_TOKEN_IDS_FILE" \
		--max-prefill-tokens "$PREFILL_MAX_TOKENS" \
		>"$PREFILL_SCHEDULE_TSV"; then
		echo "GLM52 prefill scheduler dry-run failed: $PREFILL_TOKEN_IDS_FILE" >&2
		exit 19
	fi
	final_kind="$(tail -1 "$PREFILL_SCHEDULE_TSV" | awk -F '\t' '{print $2}')"
	if [ "$final_kind" != "decode_ready" ]; then
		echo "GLM52 prefill schedule did not reach decode_ready: $PREFILL_SCHEDULE_TSV" >&2
		exit 20
	fi
	prefill_step_count="$(awk -F '\t' 'NR > 1 && $2 == "prefill" { count += 1 } END { print count + 0 }' "$PREFILL_SCHEDULE_TSV")"
	prefill_token_count="$(awk -F '\t' 'NR > 1 && $2 == "prefill" { tokens += $4 } END { print tokens + 0 }' "$PREFILL_SCHEDULE_TSV")"
	echo "glm52_local_pipeline_prefill_schedule=$PREFILL_SCHEDULE_TSV"
	echo "glm52_local_pipeline_prefill_steps=$prefill_step_count"
	echo "glm52_local_pipeline_prefill_tokens=$prefill_token_count"
}

required_cuda_link_args()
{
	local runtime_args_file
	local adapter_archive
	local backend_archive
	local table_archive
	if [ -n "${GLM52_REQUIRED_CUDA_LINK_ARGS:-}" ]; then
		printf "%s" "$GLM52_REQUIRED_CUDA_LINK_ARGS"
		return 0
	fi
	runtime_args_file="$AOT_OUTPUT_DIR/generated/runtime_link_args.txt"
	case "$runtime_args_file" in
		/*) ;;
		*) runtime_args_file="$ROOT/$runtime_args_file" ;;
	esac
	adapter_archive="$ROOT/build/modules/glm52_sm121_flashinfer_b12x_moe/libglm52_sm121_flashinfer_b12x_moe_adapter.a"
	backend_archive="$ROOT/build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_compiled_backend.a"
	table_archive="$ROOT/build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_generated_kernel_table.a"
	if [ ! -s "$adapter_archive" ] || [ ! -s "$backend_archive" ] || [ ! -s "$table_archive" ]; then
		echo "missing B12x generated archives; run setup/build before direct pipeline mode" >&2
		exit 15
	fi
	if [ ! -s "$runtime_args_file" ]; then
		echo "missing B12x runtime link args: $runtime_args_file" >&2
		exit 16
	fi
	printf "%s %s %s " "$adapter_archive" "$backend_archive" "$table_archive"
	cat "$runtime_args_file"
}

export_required_cuda_library_path()
{
	local link_args="$1"
	local arg
	local directory
	for arg in $link_args; do
		case "$arg" in
			*.so)
				directory="$(cd "$(dirname "$arg")" && pwd)"
				case ":${LD_LIBRARY_PATH:-}:" in
					*":$directory:"*) ;;
					*) export LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
				esac
				;;
		esac
	done
}

build_direct_runner()
{
	local link_args
	if [ ! -s "$MODULE_ARCHIVE" ]; then
		echo "missing GLM52 module archive for direct pipeline mode: $MODULE_ARCHIVE" >&2
		echo "build it once with make -C modules/glm52_resident_decode_stage archive" >&2
		exit 17
	fi
	link_args="$(required_cuda_link_args)"
	export_required_cuda_library_path "$link_args"
	if [ "$FORCE_RUNNER_REBUILD" = "0" ] &&
		[ -x "$DIRECT_RUNNER" ] &&
		[ "$DIRECT_RUNNER" -nt "$MODULE_ARCHIVE" ] &&
		[ "$DIRECT_RUNNER" -nt "$ROOT/modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu" ]; then
		return 0
	fi
	mkdir -p "$DIRECT_RUNNER_DIR"
	make -C "$ROOT" build/libsparkpipe_common.a build/libsparkpipe_runtime.a build/libsparkpipe_compiler.a
	"$NVCC_BIN" \
		-std=c++17 \
		-O3 \
		--use_fast_math \
		-arch="$CUDA_ARCH_VALUE" \
		-DSPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT="${ACTIVE_SEQUENCE_COUNT}u" \
		-I"$ROOT/include" \
		-I"$ROOT/modules/glm52_resident_decode_stage/include" \
		-I"$ROOT/modules/glm52_resident_decode_stage/source" \
		"$ROOT/modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu" \
		"$MODULE_ARCHIVE" \
		"$ROOT/build/libsparkpipe_runtime.a" \
		"$ROOT/build/libsparkpipe_compiler.a" \
		"$ROOT/build/libsparkpipe_common.a" \
		$link_args \
		-lcublasLt \
		-lcublas \
		-ldl \
		-o "$DIRECT_RUNNER"
}

write_input_embedding_hidden()
{
	local output_hidden="$1"
	local log_path="$OUTPUT_DIR/input_embedding_hidden.log"
	local pass_line
	local stage_env
	stage_env=(
		"NVCC=$NVCC_BIN"
		"CUDA_ARCH=$CUDA_ARCH_VALUE"
		"GLM52_MODEL_DIR=$MODEL_DIR"
		"GLM52_ALLOW_REMOTE_MODEL_DIR=0"
		"GLM52_INPUT_TOKEN_ID=$INPUT_TOKEN_ID"
		"GLM52_WRITE_INPUT_EMBEDDING_HIDDEN_BF16=1"
		"GLM52_PIPELINE_OUTPUT_HIDDEN_BF16=$output_hidden"
		"GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT=$ACTIVE_SEQUENCE_COUNT"
	)
	if [ "$RESUME" != "0" ] && [ -s "$output_hidden" ]; then
		echo "glm52_local_pipeline_input_embedding=resume output_hidden=$output_hidden"
		return 0
	fi
	build_direct_runner
	if ! (cd "$ROOT" && env "${stage_env[@]}" "$DIRECT_RUNNER" "$MAX_STAGE_US") >"$log_path" 2>&1; then
		echo "glm52_local_pipeline_input_embedding=failed log=$log_path" >&2
		tail -80 "$log_path" >&2 || true
		exit 9
	fi
	pass_line="$(stage_pass_line "$log_path")"
	if [ -z "$pass_line" ]; then
		echo "input embedding writer did not emit validation pass line: $log_path" >&2
		exit 10
	fi
	if [ ! -s "$output_hidden" ]; then
		echo "input embedding writer did not write hidden: $output_hidden" >&2
		exit 11
	fi
	echo "glm52_local_pipeline_input_embedding=passed output_hidden=$output_hidden"
}

write_input_embedding_sequence()
{
	local output_hidden="$1"
	local log_path="$OUTPUT_DIR/input_embedding_sequence.log"
	local pass_line
	local stage_env
	if [ -z "$PREFILL_TOKEN_IDS_FILE" ]; then
		echo "GLM52_LOCAL_PIPELINE_PROMPT_SEQUENCE requires GLM52_PREFILL_TOKEN_IDS_FILE" >&2
		exit 18
	fi
	stage_env=(
		"NVCC=$NVCC_BIN"
		"CUDA_ARCH=$CUDA_ARCH_VALUE"
		"GLM52_MODEL_DIR=$MODEL_DIR"
		"GLM52_ALLOW_REMOTE_MODEL_DIR=0"
		"GLM52_PREFILL_TOKEN_IDS_FILE=$PREFILL_TOKEN_IDS_FILE"
		"GLM52_WRITE_INPUT_EMBEDDING_SEQUENCE_BF16=1"
		"GLM52_PIPELINE_OUTPUT_HIDDEN_BF16=$output_hidden"
		"GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT=1"
	)
	if [ "$RESUME" != "0" ] && [ -s "$output_hidden" ]; then
		echo "glm52_local_pipeline_input_embedding_sequence=resume output_hidden=$output_hidden"
		return 0
	fi
	build_direct_runner
	if ! (cd "$ROOT" && env "${stage_env[@]}" "$DIRECT_RUNNER" "$MAX_STAGE_US") >"$log_path" 2>&1; then
		echo "glm52_local_pipeline_input_embedding_sequence=failed log=$log_path" >&2
		tail -80 "$log_path" >&2 || true
		exit 9
	fi
	pass_line="$(stage_pass_line "$log_path")"
	if [ -z "$pass_line" ]; then
		echo "input embedding sequence writer did not emit validation pass line: $log_path" >&2
		exit 10
	fi
	if [ ! -s "$output_hidden" ]; then
		echo "input embedding sequence writer did not write hidden sequence: $output_hidden" >&2
		exit 11
	fi
	echo "glm52_local_pipeline_input_embedding_sequence=passed output_hidden=$output_hidden"
}

run_stage_direct()
{
	local mode="$1"
	local first="$2"
	local count="$3"
	local input_hidden="$4"
	local output_hidden="$5"
	local pack_dir="$6"
	local log_path="$7"
	local stage_env
	stage_env=(
		"NVCC=$NVCC_BIN"
		"CUDA_ARCH=$CUDA_ARCH_VALUE"
		"GLM52_B12X_MOE_PACK_DIR=$pack_dir"
		"GLM52_MODEL_DIR=$MODEL_DIR"
		"GLM52_ALLOW_REMOTE_MODEL_DIR=0"
		"GLM52_ROUTED_CHAIN_FIRST_LAYER_INDEX=$first"
		"GLM52_ROUTED_CHAIN_LAYER_COUNT=$count"
		"GLM52_ENABLE_CUDA_GRAPH_REPLAY=$ENABLE_GRAPH_REPLAY"
		"GLM52_EXACT_PP13_STAGE_SLICE=1"
		"GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT=$ACTIVE_SEQUENCE_COUNT"
		"GLM52_PIPELINE_INPUT_HIDDEN_BF16=$input_hidden"
		"GLM52_PIPELINE_OUTPUT_HIDDEN_BF16=$output_hidden"
	)
	if [ "$mode" = "exact_final" ]; then
		stage_env+=(
			"GLM52_EXACT_PP13_STAGE_SLICE_FINAL_TOKEN=1"
		)
	elif [ "$mode" != "exact" ]; then
		echo "unknown local pipeline stage mode: $mode" >&2
		exit 8
	fi
	if ! (cd "$ROOT" && env "${stage_env[@]}" "$DIRECT_RUNNER" "$MAX_STAGE_US") >"$log_path" 2>&1; then
		echo "glm52_local_pipeline_stage=failed mode=$mode first_layer=$first layer_count=$count log=$log_path" >&2
		tail -80 "$log_path" >&2 || true
		exit 9
	fi
}

run_stage_sequence_direct()
{
	local mode="$1"
	local first="$2"
	local count="$3"
	local input_hidden="$4"
	local output_hidden="$5"
	local pack_dir="$6"
	local log_path="$7"
	local stage_env
	stage_env=(
		"NVCC=$NVCC_BIN"
		"CUDA_ARCH=$CUDA_ARCH_VALUE"
		"GLM52_B12X_MOE_PACK_DIR=$pack_dir"
		"GLM52_MODEL_DIR=$MODEL_DIR"
		"GLM52_ALLOW_REMOTE_MODEL_DIR=0"
		"GLM52_ROUTED_CHAIN_FIRST_LAYER_INDEX=$first"
		"GLM52_ROUTED_CHAIN_LAYER_COUNT=$count"
		"GLM52_ENABLE_CUDA_GRAPH_REPLAY=0"
		"GLM52_EXACT_PP13_STAGE_SEQUENCE=1"
		"GLM52_EXACT_PP13_DISABLE_GRAPH_REPLAY=1"
		"GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT=1"
		"GLM52_PIPELINE_INPUT_HIDDEN_BF16=$input_hidden"
		"GLM52_PIPELINE_OUTPUT_HIDDEN_BF16=$output_hidden"
	)
	if [ "$mode" = "exact_final" ]; then
		stage_env+=(
			"GLM52_EXACT_PP13_STAGE_SLICE_FINAL_TOKEN=1"
		)
	elif [ "$mode" != "exact" ]; then
		echo "unknown local pipeline stage mode: $mode" >&2
		exit 8
	fi
	if ! (cd "$ROOT" && env "${stage_env[@]}" "$DIRECT_RUNNER" "$MAX_STAGE_US") >"$log_path" 2>&1; then
		echo "glm52_local_pipeline_stage_sequence=failed mode=$mode first_layer=$first layer_count=$count log=$log_path" >&2
		tail -80 "$log_path" >&2 || true
		exit 9
	fi
}

run_stage_sequence()
{
	local mode="$1"
	local first="$2"
	local count="$3"
	local input_hidden="$4"
	local output_hidden="$5"
	local pack_dir="$6"
	local log_path
	local pass_line
	local total_us
	local maximum_us
	local token
	log_path="$OUTPUT_DIR/stage_sequence_${first}_${count}_${mode}.log"
	require_stage_packs "$pack_dir" "$first" "$count"
	build_direct_runner
	run_stage_sequence_direct "$mode" "$first" "$count" "$input_hidden" "$output_hidden" "$pack_dir" "$log_path"
	pass_line="$(stage_pass_line "$log_path")"
	if [ -z "$pass_line" ]; then
		echo "stage sequence did not emit validation pass line: $log_path" >&2
		exit 10
	fi
	if [ ! -s "$output_hidden" ]; then
		echo "stage sequence did not write output hidden sequence: $output_hidden" >&2
		exit 11
	fi
	if ! record_stage_summary "$mode" "$first" "$count" "$pack_dir" "$log_path"; then
		echo "stage sequence did not record summary: $log_path" >&2
		exit 12
	fi
	total_us="$(extract_field "total_us" "$pass_line")"
	maximum_us="$(extract_field "maximum_us" "$pass_line")"
	token="$(extract_field "restricted_token" "$pass_line")"
	echo "glm52_local_pipeline_stage_sequence=passed mode=$mode first_layer=$first layer_count=$count total_us=${total_us:-} maximum_us=${maximum_us:-} token=${token:-} output_hidden=$output_hidden"
}

run_stage()
{
	local mode="$1"
	local first="$2"
	local count="$3"
	local input_hidden="$4"
	local output_hidden="$5"
	local pack_dir="$6"
	local log_path
	local pass_line
	local total_us
	local maximum_us
	local token
	log_path="$OUTPUT_DIR/stage_${first}_${count}_${mode}.log"
	require_stage_packs "$pack_dir" "$first" "$count"
	build_direct_runner
	run_stage_direct "$mode" "$first" "$count" "$input_hidden" "$output_hidden" "$pack_dir" "$log_path"
	pass_line="$(stage_pass_line "$log_path")"
	if [ -z "$pass_line" ]; then
		echo "stage did not emit validation pass line: $log_path" >&2
		exit 10
	fi
	if [ ! -s "$output_hidden" ]; then
		echo "stage did not write output hidden: $output_hidden" >&2
		exit 11
	fi
	if ! record_stage_summary "$mode" "$first" "$count" "$pack_dir" "$log_path"; then
		echo "stage did not record summary: $log_path" >&2
		exit 12
	fi
	total_us="$(extract_field "total_us" "$pass_line")"
	maximum_us="$(extract_field "maximum_us" "$pass_line")"
	token="$(extract_field "restricted_token" "$pass_line")"
	echo "glm52_local_pipeline_stage=passed mode=$mode first_layer=$first layer_count=$count total_us=${total_us:-} maximum_us=${maximum_us:-} token=${token:-} output_hidden=$output_hidden"
}

run_prompt_sequence_pipeline_once()
{
	local run_label="$1"
	local first
	local count
	local mode
	local input_hidden="$OUTPUT_DIR/${run_label}_input_embedding_sequence.bf16"
	local output_hidden
	local pack_dir
	local stages=(
		"0:6:exact"
		"6:6:exact"
		"12:6:exact"
		"18:6:exact"
		"24:6:exact"
		"30:6:exact"
		"36:6:exact"
		"42:6:exact"
		"48:6:exact"
		"54:6:exact"
		"60:6:exact"
		"66:6:exact"
		"72:6:exact_final"
	)
	write_input_embedding_sequence "$input_hidden"
	for spec in "${stages[@]}"; do
		IFS=: read -r first count mode <<<"$spec"
		output_hidden="$OUTPUT_DIR/${run_label}_sequence_after_layer_$((first + count - 1)).bf16"
		pack_dir="$B12X_PACK_DIR"
		if ! pack_dir_has_stage_packs "$pack_dir" "$first" "$count"; then
			echo "missing stable B12x resident pack set for routed stage ${first}:${count}: $pack_dir" >&2
			exit 5
		fi
		if [ "$RESUME" != "0" ] && [ -s "$output_hidden" ]; then
			if ! record_stage_summary "$mode" "$first" "$count" "$pack_dir" "$OUTPUT_DIR/stage_sequence_${first}_${count}_${mode}.log"; then
				echo "resumed stage sequence did not have a reusable validation log: $first:$count" >&2
				exit 13
			fi
			echo "glm52_local_pipeline_stage_sequence=resume mode=$mode first_layer=$first layer_count=$count output_hidden=$output_hidden"
		else
			run_stage_sequence "$mode" "$first" "$count" "$input_hidden" "$output_hidden" "$pack_dir"
		fi
		input_hidden="$output_hidden"
	done
}

run_pipeline_once()
{
	local run_label="$1"
	local first
	local count
	local mode
	local input_hidden="$OUTPUT_DIR/${run_label}_input_embedding.bf16"
	local output_hidden
	local pack_dir
	local stages=(
		"0:6:exact"
		"6:6:exact"
		"12:6:exact"
		"18:6:exact"
		"24:6:exact"
		"30:6:exact"
		"36:6:exact"
		"42:6:exact"
		"48:6:exact"
		"54:6:exact"
		"60:6:exact"
		"66:6:exact"
		"72:6:exact_final"
	)
	write_input_embedding_hidden "$input_hidden"
	for spec in "${stages[@]}"; do
		IFS=: read -r first count mode <<<"$spec"
		output_hidden="$OUTPUT_DIR/${run_label}_after_layer_$((first + count - 1)).bf16"
		pack_dir="$B12X_PACK_DIR"
		if ! pack_dir_has_stage_packs "$pack_dir" "$first" "$count"; then
			echo "missing stable B12x resident pack set for routed stage ${first}:${count}: $pack_dir" >&2
			exit 5
		fi
		if [ "$RESUME" != "0" ] && [ -s "$output_hidden" ]; then
			if ! record_stage_summary "$mode" "$first" "$count" "$pack_dir" "$OUTPUT_DIR/stage_${first}_${count}_${mode}.log"; then
				echo "resumed stage did not have a reusable validation log: $first:$count" >&2
				exit 13
			fi
			echo "glm52_local_pipeline_stage=resume mode=$mode first_layer=$first layer_count=$count output_hidden=$output_hidden"
		else
			run_stage "$mode" "$first" "$count" "$input_hidden" "$output_hidden" "$pack_dir"
		fi
		input_hidden="$output_hidden"
	done
}

echo "glm52_local_pipeline_model_dir=$MODEL_DIR"
echo "glm52_local_pipeline_output_dir=$OUTPUT_DIR"
echo "glm52_local_pipeline_input_token=$INPUT_TOKEN_ID"
echo "glm52_local_pipeline_resume=$RESUME"
run_prefill_schedule
if [ "$PREFILL_ONLY" != "0" ]; then
	echo "glm52_local_pipeline_prefill_only=1"
	exit 0
fi
if [ "$PROMPT_SEQUENCE" != "0" ]; then
	run_prompt_sequence_pipeline_once "run1"
else
	run_pipeline_once "run1"
fi
final_line="$(tail -1 "$SUMMARY_TSV")"
final_token="$(printf "%s" "$final_line" | awk -F '\t' '{print $8}')"
if [ -z "$final_token" ]; then
	echo "GLM52 local pipeline did not emit a final restricted token" >&2
	exit 14
fi
echo "glm52_local_pipeline_token=$final_token"
echo "glm52_local_pipeline_summary=$SUMMARY_TSV"
