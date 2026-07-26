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
status=0
for unity in llms/*/unity.cu
do
	name=$(basename "$(dirname "$unity")")
	if "$NVCC" -std=c++17 $ARCH -O3 --use_fast_math -I. -c "$unity" -o "/tmp/lm_$name.o" 2>"/tmp/lm_$name.log"
	then
		printf "  %-14s compiled  %s bytes\n" "$name" "$(wc -c < "/tmp/lm_$name.o")"
	else
		printf "  %-14s FAILED\n" "$name"
		grep -E "error" "/tmp/lm_$name.log" | head -5
		status=1
	fi
	"$NVCC" -std=c++17 $ARCH -O3 -I. -ptx "$unity" -o "/tmp/lm_$name.ptx" 2>/dev/null || true
	target=$(grep '^.target' "/tmp/lm_$name.ptx" 2>/dev/null || echo "?")
	case "$target" in
		*sm_121a) printf "  %-14s %s\n" "" "$target" ;;
		*) printf "  %-14s WRONG TARGET: %s\n" "" "$target"; status=1 ;;
	esac
done
exit $status
