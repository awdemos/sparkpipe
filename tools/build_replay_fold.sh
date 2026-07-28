#!/bin/sh
# LmReplayFoldKernel has no model driving it yet - speculation is not wired - so
# nothing instantiates it for a device and it could stop compiling silently.
# tests/test_kda_host.py checks that it reproduces the decode state; this checks
# it still builds for a GPU at K3's real head shape.
CUDA=${CUDA_HOME:-/opt/cuda}
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA; run tools/get_cuda.sh"; exit 2; }
cd "$(dirname "$0")/.." || exit 1
cat > /tmp/lm_replay_fold.cu <<'EOF'
#include "inference/kernels/linear_attn.cuh"
template __global__ void LmReplayFoldKernel<256u, 128u, 128u>(
	uint8_t *, const uint32_t *, const LmReplayStep *, const uint32_t *,
	uint32_t, uint32_t, uint32_t);
EOF
"$CUDA/bin/nvcc" -std=c++17 -gencode arch=compute_121a,code=sm_121a -O1 -I. \
	-c /tmp/lm_replay_fold.cu -o /tmp/lm_replay_fold.o 2> /tmp/lm_replay_fold.log || {
	grep -E "error" /tmp/lm_replay_fold.log | head -5; exit 1; }
