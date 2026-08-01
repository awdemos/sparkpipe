# Phase 3 Change Manifest

Comparison baseline: `/mnt/data/sparkpipe-phase2-work` before the Phase 3 DSV4/grouped-MoE changes.

- Added: 21 files
- Modified: 27 files
- Removed: 0 files

## Added

- `docs/PHASE3_DSV4_CACHE_AND_GROUPED_MOE.md`
- `docs/PHASE3_REMAINING_WORK.md`
- `docs/PHASE3_VALIDATION_STATUS.json`
- `docs/PHASE3_VALIDATION_STATUS.md`
- `inference/llms/kimi_k3/generated_config.h`
- `inference/llms/kimi_k3/pipeline_sideband.h`
- `model-families/dsv4/include/sparkpipe/spark_dsv4_cache_arena.h`
- `model-families/dsv4/include/sparkpipe/spark_dsv4_cache_plan.h`
- `model-families/dsv4/src/spark_dsv4_cache_arena.c`
- `model-families/dsv4/src/spark_dsv4_cache_plan.c`
- `model-families/qwen36/include/sparkpipe/spark_qwen36_work_control.h`
- `model-families/qwen36/src/spark_qwen36_work_control.c`
- `model_contracts/dsv4_flash.json`
- `model_contracts/dsv4_pro.json`
- `model_contracts/k3_authoritative.json`
- `qualification/phase3_full_make_test_failure.log`
- `qualification/phase3_targeted_validation.log`
- `tests/studies/sparkpipe_dsv4_cache_plan_report.c`
- `tests/test_dsv4_cache_plan.c`
- `tests/test_grouped_moe_source_contracts.py`
- `tools/generate_k3_contract.py`

## Modified

- `Makefile`
- `examples/model_descriptions/k3_resident_decode_stage_firmware.json`
- `include/sparkpipe/spark_stage_module_common.h`
- `inference/kernels/route.cuh`
- `inference/kernels/topk.cuh`
- `inference/llms/deepseek_v4/layer.cuh`
- `inference/llms/deepseek_v4/unity.cu`
- `inference/llms/glm5_2/layer.cuh`
- `inference/llms/glm5_2/unity.cu`
- `inference/llms/kimi_k3/layer.cuh`
- `inference/llms/kimi_k3/unity.cu`
- `inference/llms/mimo_2_5/layer.cuh`
- `inference/llms/mimo_2_5/unity.cu`
- `model-families/glm52/include/sparkpipe/spark_glm52_expert_queue.h`
- `model-families/glm52/src/spark_glm52_expert_queue.c`
- `model_contracts/k3.json`
- `runtime/stage_module_common.c`
- `sources.mk`
- `tests/host_cuda/k3_layer_host.cu`
- `tests/host_cuda/k3_slice_host.cu`
- `tests/host_cuda/router_host.cu`
- `tests/studies/sparkpipe_glm52_batchplane_model.c`
- `tests/test_glm52_batch_plane.c`
- `tests/test_glm52_stage_plan.c`
- `tests/test_glm52_stagepack.c`
- `tests/test_glm52_tp_shard.c`
- `tests/test_hidden_transport.c`

## Removed

None.

