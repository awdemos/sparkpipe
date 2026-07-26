#!/bin/sh
# Every gate, each checked on its exit code.
#
# This exists because a previous run reported nine passing static_asserts from a
# translation unit that had failed to compile: the echo was not gated on the
# status. A gate that cannot fail is not a gate.
ok=0; bad=0
run() {
	if eval "$2" >/dev/null 2>&1
	then printf "  %-26s PASS\n" "$1"; ok=$((ok+1))
	else printf "  %-26s FAIL\n" "$1"; bad=$((bad+1))
	fi
}
run "ptx capability gate"  "python3 tests/test_ptx_capability_gate.py"
run "mma fragment mapping" "gcc -O2 -Wall -Wextra -o /tmp/g_f tests/test_mma_fragment_mapping.c && /tmp/g_f"
run "model constants"      "gcc -O2 -Wall -Wextra -I. -o /tmp/g_c tests/test_model_constants.c && /tmp/g_c"
run "sub-byte packing"     "gcc -O2 -Wall -Wextra -o /tmp/g_p tests/test_pack.c && /tmp/g_p"
run "free dequant"         "gcc -O2 -Wall -Wextra -o /tmp/g_d tests/test_dequant.c && /tmp/g_d"
run "reference oracle"     "gcc -O2 -Wall -Wextra -Itests -o /tmp/g_r tests/test_reference.c -lm && /tmp/g_r"
run "kv geometry"          "g++ -std=c++17 -fsyntax-only -Wall -Wextra -I. tests/test_kv_geometry.cc"
run "workspace layout"     "gcc -O2 -Wall -Wextra -I model-families/common/include -o /tmp/g_w tests/test_group_gemm_workspace.c && /tmp/g_w"
run "tensor map geometry"  "gcc -O2 -Wall -Wextra -I model-families/common/include -o /tmp/g_t tests/test_tensor_map_geometry.c && /tmp/g_t"
run "tensor map encode"    "gcc -O2 -Wall -Wextra -I model-families/common/include -I tests/cuda_driver_stub -o /tmp/g_e tests/test_tensor_map_encode.c tests/cuda_driver_stub/stub.c && /tmp/g_e"
run "autotune builds"      "gcc -O2 -Wall -Wextra -o /tmp/g_a tools/spark_lm_autotune.c"
# The real compiler, for the real target. This replaced a keyword-shim gate that
# approximated nvcc by defining the CUDA keywords away. That shim could not see a
# missing include, could not see the 48 KB static shared limit, could not see
# -arch=sm_121a dropping its own suffix, and broke outright on extern __shared__.
# A proxy for the compiler is worth having only while the compiler is
# unavailable, and tools/get_cuda.sh means it is not.
run "nvcc: sm_121a build"  "sh tools/build.sh"
printf "  ---- %d pass, %d fail\n" "$ok" "$bad"
[ "$bad" -eq 0 ]
