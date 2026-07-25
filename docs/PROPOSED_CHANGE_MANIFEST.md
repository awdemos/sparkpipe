# Proposed Change Manifest

This is a hash-based inventory, not a textual diff. Generated build products,
transient audit logs, object files, archives, Git metadata, and Python bytecode are excluded.
Retained validation receipts under `docs/` are included.

## Summary

- Added source/documentation files: **2**
- Modified source/documentation files: **23**
- Deleted source/documentation files: **0**

## Added

| Path | Bytes | SHA-256 |
|---|---:|---|
| `docs/validation-logs/host_test_suite_without_loopback.log` | 9684 | `8af54a401cdc6bdf86a16b847f0493ed156eb9cf4248b3c142a09932cd2c4526` |
| `tests/test_cuda_performance_contracts.py` | 42997 | `4a3bd34995c294948071a2aceab15b8d399cb7b6083d93fc1234a79288721104` |

## Modified

| Path | Bytes | SHA-256 |
|---|---:|---|
| `Makefile` | 58341 | `1b4c0db51b46fef0562488514e7ee6d854c2d3dda08fe9ef6c5da87df860336c` |
| `docs/VALIDATION_STATUS.json` | 4797 | `19b724d440f7cbf6afa504300267e78ec786697efa98ba1cc24e327b28f267e0` |
| `docs/VALIDATION_STATUS.md` | 3902 | `cb2df9e6b35c0194a843dad675101af1c632b19bce814604843054051a9d0a86` |
| `docs/validation-logs/host_test_suite.log` | 24052 | `d5372a2064445f57aef634c511c616c75dfcd602f9580456cbd4d74ea0fc214e` |
| `docs/validation-logs/source_tree_stability.log` | 178 | `59100e0afa26bc34a26224c2c32c4eb20b3d3664e9cbd758fd7a24151c6fbda1` |
| `model-families/common/include/sparkpipe/spark_hidden_transport.h` | 14503 | `072d254c33e3bf59f87acbdbb02e6af7d08411fda71ea23319565cee172e45b8` |
| `model-families/common/include/sparkpipe/spark_lm_fp8_tile.cuh` | 21693 | `186b36f483a8ebe8028b45cdcb9717235572963cf9ab2564305c031c21043e5b` |
| `model-families/common/include/sparkpipe/spark_lm_kernels.cuh` | 97695 | `4fa32ddf62eb2afb42273b4c2b46f40cb6b4194fffdeb2d113d85d838fdeb82e` |
| `model-families/common/src/spark_hidden_transport.c` | 36945 | `5024f9eb4b3416bc8e9b52ec3ca3a16d85d1c2ee3e56c7fb85c3c0174fc241af` |
| `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu` | 68732 | `b994d3ba7a545694b6f6dc1fd39606fd77fbb7ea45cd182fe8400e98e9f81eca` |
| `modules/glm52_resident_decode_stage/source/spark_glm52_pp13_node_context_builder_cuda.cu` | 419942 | `3cb9a0b31a7ffed1b5d90f353e10c90836f36d29b1d67c94b0825f38152a061e` |
| `modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_cuda.cu` | 15059 | `da0e682d06fc3f749ae8e0e017fe7e6a10d341a99c2c5b0611df8c4d0691d8a7` |
| `modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_linear_plan.cu` | 54009 | `2ce9f0c77a760006c1ad10a86b3ff9f02c4f9e3324ed8b3b29b2271a8d3132e4` |
| `modules/glm52_resident_decode_stage/source/spark_glm52_sm121_required_decode_stage.cu` | 1035669 | `50b2b1dc2c8ec9e5bdf73fe37c7c76522ba859df70395d988860d208f97d56fc` |
| `modules/hidden_transport_spark_host_rdma_verbs.cu` | 129007 | `350c1d5bef0ff6541e597390bca644bd362ca31a2ba4e9d54d188032eb9858b1` |
| `modules/k3_resident_decode_stage/include/sparkpipe/spark_k3_resident_decode_stage_firmware.h` | 19411 | `e6608987fb65404b0809cc0bd597ca1ed1ae07721e74d6cd9d9801f50dc24e28` |
| `modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu` | 92526 | `ba037e54fa82662d351c742233679032313abbef9936be100abd26092742fb58` |
| `modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_module.c` | 103110 | `d3f8c8b10988acefbd60ce249533fe2c81b3ef58f5292be630a4ebe051656e1d` |
| `modules/mimo25_resident_decode_stage/source/spark_mimo25_resident_decode_stage_cuda.cu` | 24106 | `652c586ae94275fbf6b2f9ade64a71631149411c7a2fb52cc0e0d7a201f44877` |
| `modules/qwen36_resident_decode_stage/source/spark_qwen36_resident_decode_stage_cuda.cu` | 60972 | `baf117b47cff84542901b372d9baae7621606256833ba110aa73ba1e8cc1556f` |
| `modules/qwen36_resident_decode_stage/source/spark_qwen36_resident_decode_stage_module.c` | 107819 | `71b4d3fcc25ba82fe933bd72aba88288adf495b3ec4e9c2e892f499f5ea51b60` |
| `tests/test_glm52_exact_pp13_prefill_hidden.py` | 76596 | `288b7a9007c5ef29bf5eb39073401ea92056e7d095d95ad5a53dcc19d3ec785a` |
| `tests/test_glm52_quantized_cuda_contract.py` | 10246 | `31062815fd8968a9f010d473588723d2796128f26ffa9ff2bf5f371b432b5827` |

## Deleted from Proposed Tree

None.
