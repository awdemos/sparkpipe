#!/bin/sh
# Compile every model's unity build for the real target.
#
# THE ARCH FLAG MATTERS AND IS NOT THE OBVIOUS ONE. -arch=sm_121a emits
# .target sm_121 - it drops the architecture-specific suffix, and then every
# instruction this codebase depends on fails to assemble: the FP4 conversions,
# the block-scaled mma, all of it. -gencode arch=compute_121a,code=sm_121a is
# the form that keeps it. The build asserts the emitted target rather than
# trusting the flag.
set -e
CUDA=${CUDA_HOME:-/opt/cuda}
ARCH="-gencode arch=compute_121a,code=sm_121a"
NVCC="$CUDA/bin/nvcc"
[ -x "$NVCC" ] || { echo "no nvcc at $NVCC; run tools/get_cuda.sh"; exit 2; }
# Every model family's headers, so a driver that includes its model header
# compiles regardless of which family it belongs to. Keep in step with
# tools/cuda13_sm121a_compile_gate.sh's include list.
INCLUDES="-I. -Iinclude -Ideployment/include \
	-Imodel-families/glm52/include -Imodel-families/qwen36/include \
	-Imodel-families/dsv4/include -Imodel-families/k3/include \
	-Imodel-families/mimo25/include \
	-Imodules/glm52_resident_decode_stage/include"
status=0
for unity in inference/llms/*/unity.cu
do
	name=$(basename "$(dirname "$unity")")
	if "$NVCC" -std=c++17 $ARCH -O3 --use_fast_math $INCLUDES -c "$unity" -o "/tmp/lm_$name.o" 2>"/tmp/lm_$name.log"
	then
		printf "  %-14s compiled  %s bytes\n" "$name" "$(wc -c < "/tmp/lm_$name.o")"
	else
		printf "  %-14s FAILED\n" "$name"
		grep -E "error" "/tmp/lm_$name.log" | head -5
		status=1
	fi
	# The binder, where a model has one: it maps the host's weight struct to the
	# layer's buffers and is where a wrong field name produces fluent output.
	if [ -f "$(dirname "$unity")/bind.cu" ]
	then
		if "$NVCC" -std=c++17 $ARCH -O1 -I. \
			-Imodules/glm52_resident_decode_stage/include -Iinclude \
			-Ideployment/include -Imodel-families/glm52/include \
			-c "$(dirname "$unity")/bind.cu" -o "/tmp/lm_${name}_bind.o" 2>"/tmp/lm_${name}_bind.log"
		then
			printf "  %-14s bind      %s bytes\n" "" "$(wc -c < "/tmp/lm_${name}_bind.o")"
		else
			printf "  %-14s BIND FAILED\n" ""
			grep -E "error" "/tmp/lm_${name}_bind.log" | head -3
			status=1
		fi
	fi
	"$NVCC" -std=c++17 $ARCH -O3 $INCLUDES -ptx "$unity" -o "/tmp/lm_$name.ptx" 2>/dev/null || true
	target=$(grep '^.target' "/tmp/lm_$name.ptx" 2>/dev/null || echo "?")
	case "$target" in
		*sm_121a) printf "  %-14s %s\n" "" "$target" ;;
		*) printf "  %-14s WRONG TARGET: %s\n" "" "$target"; status=1 ;;
	esac
done
# The production decode stage, which is not in llms/ yet and is exactly what a
# gate that only covers the new tree cannot see. Two deletions have broken it
# already - a header removed as superseded, and a transport removed alongside
# the kernel library it shared a directory with - and both times every gate was
# green.
# The serving adapter, both ways. It is the seam: legacy calls
# LaunchStageSlice, first-party calls Glm52StageSlice, and both must compile
# because a build that only checks one lets the other rot silently.
# The serving adapter: the seam between the scheduler's frames and the kernels.
ADAPTER=inference/stage/serving_adapter.cu
if "$NVCC" -std=c++17 $ARCH -O1 -I. \
	-Imodules/glm52_resident_decode_stage/include -Iinclude \
	-Ideployment/include -Imodel-families/glm52/include \
	-c "$ADAPTER" -o /tmp/lm_adapter.o 2>/tmp/lm_adapter.log
then
	printf "  %-14s compiled  %s bytes\n" "adapter" "$(wc -c < /tmp/lm_adapter.o)"
else
	printf "  %-14s FAILED\n" "adapter"
	grep -E "error" /tmp/lm_adapter.log | head -3
	status=1
fi

exit $status
