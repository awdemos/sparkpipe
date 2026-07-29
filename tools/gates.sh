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
run "weight binding"       "gcc -O2 -Wall -Wextra -I. -Imodules/glm52_resident_decode_stage/include -Iinclude -Ideployment/include -Imodel-families/glm52/include -o /tmp/g_b tests/test_pack_bind.c && /tmp/g_b"
run "sidebands"            "gcc -O2 -Wall -Wextra -I. -o /tmp/g_s tests/test_sideband.c && /tmp/g_s"
run "kv cache"             "gcc -O2 -Wall -Wextra -I. -o /tmp/g_kv tests/test_cache.c && /tmp/g_kv"
run "kv geometry"          "g++ -std=c++17 -fsyntax-only -Wall -Wextra -I. tests/test_kv_geometry.cc"
run "workspace layout"     "gcc -O2 -Wall -Wextra -I. -o /tmp/g_w tests/test_group_gemm_workspace.c && /tmp/g_w"
run "tensor map geometry"  "gcc -O2 -Wall -Wextra -I. -o /tmp/g_t tests/test_tensor_map_geometry.c && /tmp/g_t"
run "tensor map encode"    "gcc -O2 -Wall -Wextra -I. -Itests/cuda_driver_stub -o /tmp/g_e tests/test_tensor_map_encode.c tests/cuda_driver_stub/stub.c && /tmp/g_e"
# The real compiler, for the real target. This replaced a keyword-shim gate that
# approximated nvcc by defining the CUDA keywords away. That shim could not see a
# missing include, could not see the 48 KB static shared limit, could not see
# -arch=sm_121a dropping its own suffix, and broke outright on extern __shared__.
# A proxy for the compiler is worth having only while the compiler is
# unavailable, and tools/get_cuda.sh means it is not.
run "launch planning"      "g++ -std=c++17 -O2 -Wall -Wextra -I. -D__host__= -D__device__= -o /tmp/g_l tests/test_launch.c && /tmp/g_l"
run "config coverage"      "python3 tests/test_config_coverage.py"
# Carried forward from #514, whose patch targeted a file the rewrite deleted.
# The hazard survived the rewrite with a different failure mode: the old tile
# staged out-of-row data past the K bound, the new one drops the tail via
# k_tiles = input_dimension / TILE_K. Both are wrong output with no crash.
run "gemm K alignment"     "python3 tests/test_gemm_k_alignment.py"
run "rope pairing"         "python3 tests/test_rope_pairing.py"
run "layer kinds"          "python3 tests/test_layer_kinds.py"
run "situ activation"      "python3 tests/test_situ_activation.py"
run "kda decay bound"      "python3 tests/test_kda_decay.py"
run "kernel launches"      "python3 tests/test_kernel_launches.py"
run "mla absorption"       "python3 tests/test_mla_absorption.py"
run "expert grouping"      "python3 tests/test_expert_grouping.py"
# The real kernels, run on a CPU. Not a reimplementation: kda_host.cu includes
# inference/kernels/linear_attn.cuh unmodified and gives it a grid. Reverting
# either of the two bugs this path had - the undecayed prediction, the dropped
# dt_bias - takes the relative error from 2e-3 to 3e-1 and 6e-2.
run "kda on host"          "python3 tests/test_kda_host.py"
# The routing path had three defects, all found by reading and none by running.
# Emitting the biased score as the weight produces 9 failures here; skipping the
# renormalisation produces 17.
run "router on host"       "python3 tests/test_router_host.py"
# Six more kernels the other two harnesses do not reach, including the MoE
# finalize whose launch was wrong four ways and compiled.
run "layer on host"        "python3 tests/test_layer_host.py"
# The MLA store and attention over a paged cache, two sequences with interleaved
# pages so ignoring the page table is visible.
run "mla on host"          "python3 tests/test_mla_host.py"
# Dataflow, not arithmetic. Every per-kernel harness passes and an audit still
# found three defects in which buffer feeds which kernel. Reintroducing the
# shared-expert overwrite makes this fail.
run "layer dataflow"       "python3 tests/test_layer_dataflow.py"
# The checkpoint quantises the routed experts and nothing else, because only
# they saw quantisation-aware training. Putting attention back on Format fails.
run "k3 quant recipe"      "python3 tests/test_k3_quant_recipe.py"
# A whole layer, executed. Found a divide-by-zero in production code on its
# first successful run, and catches the shared-expert overwrite by seeing the
# routed value missing from the output rather than by reading the source.
run "k3 layer on host"     "python3 tests/test_k3_layer_host.py"
run "k3 slice on host"     "python3 tests/test_k3_slice_host.py"
run "k3 engine on host"    "python3 tests/test_k3_engine.py"
run "k3 pack"              "python3 tests/test_k3_pack.py"
run "k3 tp shard"          "python3 tests/test_k3_shard.py"
run "k3 shard table"       "python3 tests/test_k3_shard_table.py"
run "k3 stage doorway"     "gcc -Iinclude -Imodel-families/k3/include -Imodel-families/glm52/include -DNDEBUG -c modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_validation.c -o /tmp/g_k3v.o"
run "stage module + model"  "gcc -Iinclude -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source -Imodel-families/glm52/include -DNDEBUG -c inference/stage/module.c -o /tmp/g_mod.o && gcc -Iinclude -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source -Imodel-families/glm52/include -DNDEBUG -c modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_validation.c -o /tmp/g_val.o"
run "k3 kv seam"          "gcc -O2 -Wall -Wextra -Iinclude -Imodel-families/glm52/include -Imodel-families/k3/include -o /tmp/g_k3kv tests/test_k3_kv_cache.c cache/kv_cache.c && /tmp/g_k3kv"
run "state pool"          "gcc -O2 -Wall -Wextra -I. -o /tmp/g_sp tests/test_state_pool.c && /tmp/g_sp"
run "k3 kv geometry"      "python3 tests/test_k3_kv_geometry.py"
run "fast defaults"        "python3 tests/test_fast_defaults.py"
run "code size"           "python3 tests/test_code_size.py"
run "dry naming law"       "python3 tests/test_dry_law.py"
# The grouped selection path has no model in this tree, so nothing instantiates
# it and nothing would notice it failing to compile.
run "grouped topk builds"  "sh tools/build_grouped_topk.sh"
run "replay fold builds"   "sh tools/build_replay_fold.sh"
run "kernel algorithms"    "python3 tests/test_kernel_algorithms.py"
run "model contracts"      "python3 tests/test_model_driver_contracts.py"
run "nvcc: sm_121a build"  "sh tools/build.sh"
# The Makefile, which no gate covered. It did not parse: the reorganisation moved
# twelve sources and $(patsubst src/%.c,...) returned the non-matching paths
# UNCHANGED, so runtime/filesystem.c reached -include and make read a C file as a
# makefile. Every other gate was green throughout. .updaterepo-policy names four
# make targets as its validation and none of them could run.
run "makefile parses"      "make -n all"
run "makefile: test"       "make -n test"
run "makefile: tools"      "make -n tools"
run "makefile: backend"    "make -n glm52_ring_service_backend"
run "every source exists"  "python3 tests/test_sources_exist.py"
printf "  ---- %d pass, %d fail\n" "$ok" "$bad"
[ "$bad" -eq 0 ]
