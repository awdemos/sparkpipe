# Validation Status

Generated at `2026-07-24T09:51:42.134230+00:00`.

This receipt used bounded sequential partitions because the execution sandbox terminates commands longer than approximately 15 seconds. The C binary and Python test inventories were read from the Makefile, executed exactly once in order, and verified complete before this receipt was issued.

This is a host-side receipt for the audited source tree. It does not qualify CUDA execution, numerical correctness, latency, throughput, or production readiness.

The harness acquires an exclusive repository lock, fingerprints the source tree, and begins with `make clean`, so a passing receipt cannot be inherited from an incremental or concurrent build. The final fingerprint must match before the receipt is issued.

Input source fingerprint: `3b40fac339929ca22efd36bce35ab046c194f1c186e65860ac32d62474cc1f78` (5030 files).

Output source fingerprint: `3b40fac339929ca22efd36bce35ab046c194f1c186e65860ac32d62474cc1f78` (5030 files).

| Step | Category | Status | Return code | Duration | Log |
|---|---|---:|---:|---:|---|
| `clean_build` | preflight | **passed** | 0 | 0.032s | `docs/validation-logs/clean_build.log` |
| `python_tool_syntax` | tooling | **passed** | 0 | 0.569s | `docs/validation-logs/python_tool_syntax.log` |
| `host_build` | build | **passed** | 0 | 7.357s | `docs/validation-logs/host_build.log` |
| `core_boundary_audit` | architecture | **passed** | 0 | 0.568s | `docs/validation-logs/core_boundary_audit.log` |
| `non_glm_model_driver_contracts` | driver-contract | **passed** | 0 | 1.226s | `docs/validation-logs/non_glm_model_driver_contracts.log` |
| `memory_contracts` | contract | **passed** | 0 | 4.129s | `docs/validation-logs/memory_contracts.log` |
| `host_test_prerequisites` | test | **passed** | 0 | 0.745s | `docs/validation-logs/host_test_prerequisites.log` |
| `host_c_tests_01_of_08` | test | **passed** | 0 | 0.360s | `docs/validation-logs/host_c_tests_01_of_08.log` |
| `host_c_tests_02_of_08` | test | **passed** | 0 | 3.925s | `docs/validation-logs/host_c_tests_02_of_08.log` |
| `host_c_tests_03_of_08` | test | **passed** | 0 | 0.587s | `docs/validation-logs/host_c_tests_03_of_08.log` |
| `host_c_tests_04_of_08` | test | **passed** | 0 | 0.300s | `docs/validation-logs/host_c_tests_04_of_08.log` |
| `host_c_tests_05_of_08` | test | **passed** | 0 | 1.436s | `docs/validation-logs/host_c_tests_05_of_08.log` |
| `host_c_tests_06_of_08` | test | **passed** | 0 | 0.834s | `docs/validation-logs/host_c_tests_06_of_08.log` |
| `host_c_tests_07_of_08` | test | **passed** | 0 | 1.021s | `docs/validation-logs/host_c_tests_07_of_08.log` |
| `host_c_tests_08_of_08` | test | **passed** | 0 | 2.279s | `docs/validation-logs/host_c_tests_08_of_08.log` |
| `host_python_tests_01_of_09` | test | **passed** | 0 | 4.786s | `docs/validation-logs/host_python_tests_01_of_09.log` |
| `host_python_tests_02_of_09` | test | **passed** | 0 | 5.611s | `docs/validation-logs/host_python_tests_02_of_09.log` |
| `host_python_tests_03_of_09` | test | **passed** | 0 | 1.720s | `docs/validation-logs/host_python_tests_03_of_09.log` |
| `host_python_tests_04_of_09` | test | **passed** | 0 | 3.478s | `docs/validation-logs/host_python_tests_04_of_09.log` |
| `host_python_tests_05_of_09` | test | **passed** | 0 | 4.680s | `docs/validation-logs/host_python_tests_05_of_09.log` |
| `host_python_tests_06_of_09` | test | **passed** | 0 | 2.126s | `docs/validation-logs/host_python_tests_06_of_09.log` |
| `host_python_tests_07_of_09` | test | **passed** | 0 | 1.613s | `docs/validation-logs/host_python_tests_07_of_09.log` |
| `host_python_tests_08_of_09` | test | **passed** | 0 | 6.157s | `docs/validation-logs/host_python_tests_08_of_09.log` |
| `host_python_tests_09_of_09` | test | **passed** | 0 | 5.509s | `docs/validation-logs/host_python_tests_09_of_09.log` |
| `required_host_targets` | build | **passed** | 0 | 0.016s | `docs/validation-logs/required_host_targets.log` |
| `cuda_node_context_builder` | optional-cuda-build | **skipped** | — | 0.000s | `docs/validation-logs/cuda_node_context_builder.log` |
| `source_tree_stability` | preflight | **passed** | 0 | 0.315s | `docs/validation-logs/source_tree_stability.log` |

## Commands

### `clean_build`

```sh
make clean
```

### `python_tool_syntax`

```sh
/opt/pyvenv/bin/python3 -m py_compile tools/audit_core_boundaries.py tools/generate_proposed_change_manifest.py tools/package_audited_proposal.py tools/run_deep_audit_validation.py tests/test_model_driver_contracts.py
```

### `host_build`

```sh
make -j2 all
```

### `core_boundary_audit`

```sh
make architecture_audit
```

### `non_glm_model_driver_contracts`

```sh
make model_driver_contracts
```

### `memory_contracts`

```sh
/opt/pyvenv/bin/python3 tests/test_memory_contracts.py
```

### `host_test_prerequisites`

```sh
make -j2 build/sparkpipe_glm52_prefill_dryrun glm52_w8lut_quality_reference
```

### `host_c_tests_01_of_08`

```sh
bounded-c-test-group build/test_json build/test_hidden_transport build/test_memlink build/test_release build/test_glm52_kv_cache
```

Note: binaries were built with make and then executed sequentially

### `host_c_tests_02_of_08`

```sh
bounded-c-test-group build/test_kv_store build/test_kv_mooncake build/test_qwen36_work_control build/test_glm52_dspark build/test_glm52_stage_plan
```

Note: binaries were built with make and then executed sequentially

### `host_c_tests_03_of_08`

```sh
bounded-c-test-group build/test_glm52_mtp_tree build/test_glm52_tp_shard build/test_glm52_shape_config build/test_tp_collective build/test_glm52_row_allocator
```

Note: binaries were built with make and then executed sequentially

### `host_c_tests_04_of_08`

```sh
bounded-c-test-group build/test_glm52_stagepack build/test_glm52_production_topology build/test_glm52_pp13_runtime build/test_glm52_cuda_resident_ipc build/test_glm52_cuda_resident_gate
```

Note: binaries were built with make and then executed sequentially

### `host_c_tests_05_of_08`

```sh
bounded-c-test-group build/test_glm52_pp13_work_control build/test_glm52_scheduler build/test_glm52_prefix_cache build/test_glm52_request_api build/test_glm52_long_context
```

Note: binaries were built with make and then executed sequentially

### `host_c_tests_06_of_08`

```sh
bounded-c-test-group build/test_tokenizer build/test_glm52_prompt_pipeline build/test_glm52_serving_engine build/test_glm52_service build/test_glm52_service_backend
```

Note: binaries were built with make and then executed sequentially

### `host_c_tests_07_of_08`

```sh
bounded-c-test-group build/test_glm52_compat_api build/test_glm52_http_gateway build/test_glm52_pp13_rank_daemon build/test_model_description build/test_stage_module_common
```

Note: binaries were built with make and then executed sequentially

### `host_c_tests_08_of_08`

```sh
bounded-c-test-group build/test_module_library build/test_driver_compiler build/test_orchestrator build/test_glm52_resident_decode_stage_firmware build/test_glm52_resident_decode_stage_production_runner
```

Note: binaries were built with make and then executed sequentially

### `host_python_tests_01_of_09`

```sh
bounded-python-test-group tests/test_api_stress.py tests/test_memory_contracts.py
```

Note: Python tests were executed sequentially in Makefile order

### `host_python_tests_02_of_09`

```sh
bounded-python-test-group tests/test_b12x_scale_layout.py tests/test_glm52_dspark_manifest.py tests/test_glm52_dspark_artifact_preflight.py
```

Note: Python tests were executed sequentially in Makefile order

### `host_python_tests_03_of_09`

```sh
bounded-python-test-group tests/test_glm52_dspark_trace_quality.py tests/test_glm52_b12x_pack_worker.py tests/test_glm52_b12x_resident_manifest.py
```

Note: Python tests were executed sequentially in Makefile order

### `host_python_tests_04_of_09`

```sh
bounded-python-test-group tests/test_glm52_fp8_pack_layout.py tests/test_glm52_w8lut_pack_layout.py tests/test_glm52_w8lut_artifact_preflight.py
```

Note: Python tests were executed sequentially in Makefile order

### `host_python_tests_05_of_09`

```sh
bounded-python-test-group tests/test_glm52_nvfp4_artifact_preflight.py tests/test_glm52_quantized_cuda_contract.py tests/test_glm52_w8lut_ring_preflight.py
```

Note: Python tests were executed sequentially in Makefile order

### `host_python_tests_06_of_09`

```sh
bounded-python-test-group tests/test_glm52_w8lut_validation_wiring.py tests/test_glm52_w8lut_stage_pack_watch.py tests/test_glm52_b12x_relocate_aot_bundle.py
```

Note: Python tests were executed sequentially in Makefile order

### `host_python_tests_07_of_09`

```sh
bounded-python-test-group tests/test_glm52_b12x_deterministic_finalize.py tests/test_glm52_final_from_hidden_mode.py tests/test_glm52_exact_pp13_prefill_hidden.py
```

Note: Python tests were executed sequentially in Makefile order

### `host_python_tests_08_of_09`

```sh
bounded-python-test-group tests/test_glm52_firmware_package.py tests/test_measured_status.py tests/test_release_assemble.py
```

Note: Python tests were executed sequentially in Makefile order

### `host_python_tests_09_of_09`

```sh
bounded-python-test-group tests/test_glm52_stage_pack.py tests/test_glm52_stage_bucket_sweep.py tests/test_glm52_prompt_pipeline_input.py
```

Note: Python tests were executed sequentially in Makefile order

### `required_host_targets`

```sh
make -j2 glm52_pp13_service_backend tools
```

### `cuda_node_context_builder`

```sh
make -j2 glm52_pp13_node_context_builder
```

Note: nvcc is unavailable; this is not a pass

### `source_tree_stability`

```sh
internal source-tree-fingerprint
```

Note: input=3b40fac339929ca22efd36bce35ab046c194f1c186e65860ac32d62474cc1f78 output=3b40fac339929ca22efd36bce35ab046c194f1c186e65860ac32d62474cc1f78 input_files=5030 output_files=5030

## Interpretation

- `passed` means only that the named host command returned success.
- `failed` is a release blocker for this proposal.
- `skipped` is not a pass; the required local tool was unavailable or a preflight blocked the step.
- Non-GLM drivers remain `NOT_MEASURED` until exact-hardware GPU receipts exist.
