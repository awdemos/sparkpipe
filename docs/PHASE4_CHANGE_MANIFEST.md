# Proposed Change Manifest

This is a hash-based inventory, not a textual diff. Generated build products,
transient audit logs, object files, archives, Git metadata, and Python bytecode are excluded.
Retained validation receipts under `docs/` are included.

## Summary

- Added source/documentation files: **20**
- Modified source/documentation files: **82**
- Deleted source/documentation files: **14**

## Added

| Path | Bytes | SHA-256 |
|---|---:|---|
| `docs/PHASE4_FOUNDATIONAL_CORRECTNESS.md` | 3525 | `128aca4dcfff4c043648b4dade5632aec08a0dd28d6ec1351ab1b182f8562edc` |
| `docs/PHASE4_REMAINING_WORK.md` | 10454 | `2d2a6945ad65c40584c706d2e57aaba11be580ef14f06e2e7c5b9863c7ceffd6` |
| `docs/PHASE4_SHARED_CUDA_AND_PRECISION.md` | 4271 | `6174b4a59261902b5252b05ca51de42485e988b1fe79176ba936ca8ba73168c0` |
| `docs/PHASE4_VALIDATION_STATUS.json` | 2230 | `eda083cc862a22f56b012165b1ee7ca1c43b6fd3fbeb0e577aba73c5291176bf` |
| `docs/PHASE4_VALIDATION_STATUS.md` | 3783 | `ecf2e7f387f11ca39b9536e3b8922769abb52f2a7e9e3d5e7c2cff03e1d334f1` |
| `inference/kernels/scale.cuh` | 8393 | `e969110dcf7280a23730154b4a2f22f87dfc345eadf3ba8f647d5c2a92e95544` |
| `inference/llms/deepseek_v4_pro/unity.cu` | 3508 | `cdddfc64894c6887ddd30e74b1b13d2e769b7f0047fbf28e97f77910aa23029c` |
| `qualification/phase4/architecture_audit.log` | 306 | `4c8d4400d95b5d8aa6d06ca5bb1f77ea418630f45e84f35221d60b1e177e4c21` |
| `qualification/phase4/cuda_install_attempt.log` | 1757 | `e66ab0129ee952e3406b2a2ed545f89674035e2ff9d8789b364f5291205a2e06` |
| `qualification/phase4/gates.log` | 2487 | `3e27fbb682db01a4f257f0e7b86a3636af59d6bad42fb7b96677cba2a945e4b4` |
| `qualification/phase4/host_make_all.log` | 9790 | `0055e3fc493f5cdc8926afd160089ea727dd061f159362b3280d335e24117d3d` |
| `qualification/phase4/host_make_test.log` | 40991 | `c69aa36e938c3ee1931cbb3ece0b786c8f57caa5b3740eb7d90934cca5918beb` |
| `runtime/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runtime/pack/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `tests/test_cuda_performance_contracts.py` | 12323 | `bcfa712f5d91acc07fa28d739a6a16b7ef884dc9e7726f55a27507987ba611ee` |
| `tests/test_glm52_unity_precision_contract.py` | 1952 | `04bf86cb8d0f80f76ce94a802f814e42952ed823652e2aa64d455b3bd176cefe` |
| `tests/test_python_syntax.py` | 825 | `8fb5b83411b7446689885fd1c62ece7436e070021fb6bbba19380c3248364aba` |
| `tests/test_qwen36_bf16_contract.py` | 2270 | `b898fd73d97905bef53330bf37254d13821cadf21ec1ef69557c5e8ff581ed3e` |
| `tools/audit_core_boundaries.py` | 10064 | `b7372782598406e91d1e1e22b68d7f9cc0e4780d5a3a18b355ca9c66fc4cbf5c` |
| `tools/glm52_spark2_local_pipeline_gate.sh` | 3359 | `e2ea13714fab59adf196d266773b538a47ef1b2f5542c751b9b8b066d275d92a` |

## Modified

| Path | Bytes | SHA-256 |
|---|---:|---|
| `.github/workflows/cuda13-sm121a-compile.yml` | 1169 | `d2d27b6a52d8c3bd44602506683a98e2c3d45f9695830524e667f54c0fb3c052` |
| `HANDOFF.md` | 3743 | `0288922a5956ae6e0ed8124e8fc48d6ca9d89490e1ca1db0a2fbcdf9073406e4` |
| `Makefile` | 58868 | `ae5e865a10b5eafa540d661c912d6a16ef3a5454bd72757a68d87dff0c232eac` |
| `README.md` | 10536 | `e9ea2c59ac9972fcfcb18e2da2513ae2d86fc9db8aecbb62da14724ea1c176bb` |
| `STATUS.md` | 1444 | `608d52b2f0cf18e78eb453c617dd3795f45405a7823f00d596ec4c764258ed8e` |
| `api/service.c` | 48567 | `24edeac4376bba31599de57c0713d7abcc1149221808d38a364f78cb96b73599` |
| `api/serving_engine.c` | 81239 | `0da66e1a44266da1cde66be11a99a226c0588db39a47c25f5d577fdeb0f7c4ac` |
| `docs/GLM52_PP13_WRONG_TOKENS_ROOT_CAUSE_20260710.md` | 6854 | `7453bdf5756244dafd4f45c77cff67d728b4923ef1a13394bbbf2930097a3938` |
| `docs/VALIDATION_STATUS.json` | 2230 | `eda083cc862a22f56b012165b1ee7ca1c43b6fd3fbeb0e577aba73c5291176bf` |
| `docs/VALIDATION_STATUS.md` | 3783 | `ecf2e7f387f11ca39b9536e3b8922769abb52f2a7e9e3d5e7c2cff03e1d334f1` |
| `docs/techdebt.md` | 13786 | `d131d99a5e85bcb13f8d59a876519a079cde67fde117108936c9809ac6ad6f42` |
| `examples/model_descriptions/glm52_resident_decode_stage_firmware.json` | 5968 | `02038c09298a4eaa0765742204ac09c160f4caa4b1ac617659cd081ed5d031b9` |
| `include/sparkpipe/spark_status.h` | 2722 | `b4ca7b4aeaed6486f99abad1ccaafb427952fe2c268e5e95ab264ab563604be8` |
| `inference/kernels/attn.cuh` | 24438 | `95d3c36b1e10968c333490a8bc990f7b80655007ae07c018b71e121a7f489680` |
| `inference/kernels/formats/bf16.cuh` | 2089 | `1a654eaa2aa8fd4b796dde00fdd61ff2e381d85fb4bff13589e44dc2b95a9c00` |
| `inference/kernels/gemm.cuh` | 17002 | `151d38108d214829d2b0ee224900a7063b43decbef953a3a6ccb3bfdde864c63` |
| `inference/kernels/layout.cuh` | 3074 | `26bd2762a676acdfdbbd321624bc04d3767c7d5f755728a3ff5e2a1b20204ef4` |
| `inference/kernels/mma.cuh` | 17279 | `1657c77c5b409ef3de06836a6b75255829998ed1564e688f99b231e1bbed3441` |
| `inference/kernels/norm.cuh` | 24737 | `152519d59f7fbf38730bc68ced3e376b060030d6ab0b56b423e162e958e73d90` |
| `inference/kernels/project.cuh` | 20091 | `3e352430780ee8cd39deec91b1c2235e17f1d94670dda716dcb523141c8c1316` |
| `inference/kernels/tensor_map.cuh` | 5785 | `5e95ae865eb4e5290892d89d3717db8ead8be1b5c077abf0e40b9fef3e971661` |
| `inference/kernels/tile.cuh` | 8581 | `77bd60ed5922fc92033d3faf03c74acf06ef4cc9f731a4257bffe4bd058469d5` |
| `inference/kernels/tma.cuh` | 7823 | `2c991ae6954360faf45af5d42f648bf4afcc9b589aa8b7c4c176e8faa39d4bb2` |
| `inference/llms/deepseek_v4/layer.cuh` | 20793 | `ef3dbdae3ca513f6ac829d797c4dc5dca85e6a18b20ca0226c71d49f77e9a658` |
| `inference/llms/deepseek_v4/unity.cu` | 6269 | `cf3b0316090a5bfe872789eea050cef6650cbb843516bb84e8abc521ed67edd1` |
| `inference/llms/deepseek_v4_pro/config.h` | 1165 | `3f5de5fda72d37f9a6d36c34656e0cb242c6f1adf7ebf09a56c0c08b92081e88` |
| `inference/llms/glm5_2/api.h` | 867 | `3f61053a43194bc3839b82bdba91e698c7c6ea5d82d6daa3b281c36f58e35136` |
| `inference/llms/glm5_2/bind.cu` | 6994 | `6290ebff5d43a16a91aafc7847800c49b27630f33fdb79e2aff49d170cc26f5a` |
| `inference/llms/glm5_2/config.h` | 6825 | `2f7bb7a7347336b0f0fb2c852d288e40ecec8edde0542fad4e5984c61c4e2c8a` |
| `inference/llms/glm5_2/layer.cuh` | 19375 | `a1d1f4bff3d34a26136bf8d85cc80d4836f94f430d444d7fd4f8e0a9d8501908` |
| `inference/llms/glm5_2/unity.cu` | 6856 | `ecf662c31fcc7d9820649b40db4a1edb273794bb819de4084836235f700d1207` |
| `inference/llms/kimi_k3/layer.cuh` | 42357 | `e3c7fd565f45a731a7ee98b65e39e860d91f483817cb7559063b2440587ac2b1` |
| `inference/llms/kimi_k3/slice.cuh` | 22153 | `df42cb258f1c9d1e879a62f0a445880c53cfa7a5fcd1ed3867c599837321643f` |
| `inference/llms/kimi_k3/unity.cu` | 8917 | `c160c75907e40adccf81a46f6c196adebcd6e8fbd4876a2c150fe8d554df5915` |
| `inference/llms/mimo_2_5/layer.cuh` | 16151 | `45956b3e75b3d51ff3eaf9c69fe68ca1a638a84c38edeb123bf4a4a63b65d2e6` |
| `inference/llms/mimo_2_5/unity.cu` | 9815 | `e647677053b9cbdfdcd931b0b7989b5eee5f40d4aa5c8ef3c26090e495714635` |
| `inference/llms/qwen_3_6/bind.cu` | 6136 | `5450dfea1e02e274511247a7d99f8c70807a28e07c59748d26acf6048cd0821a` |
| `inference/llms/qwen_3_6/layer.cuh` | 16009 | `a50b0a1dec0c5563e939d51bee6ad8c78df5ef148b5aac5e23126d6f31f3a4a2` |
| `inference/llms/qwen_3_6/unity.cu` | 6601 | `0975ec8effc36dd3aba704d2a1ff66dd733e52c27a29d1d05977dcb9e2a9396c` |
| `model-families/glm52/include/sparkpipe/spark_glm52_dspark.h` | 12396 | `3d41025800696c0e5e8ab53204b7d2327d31fafa9c5c24108ffe882459c47010` |
| `model-families/glm52/include/sparkpipe/spark_glm52_kv_cache.h` | 8475 | `3276bb48113a7a0f29f222d51f2a6173bae0715a0b4c97bb8367f4c9de4e5b95` |
| `modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h` | 26601 | `2e821ab06de4943cfbf95d51e04584aad4c14ec4e9f5bb7ef233c9ef53cf69fe` |
| `node/rank_runtime.c` | 32678 | `841bc3fcdb68b446e10c796fc9c15ae171dfe333f76dd945de5e0cf7003289c1` |
| `runtime/gemm.cuh` | 13022 | `2784e04dbe6905ad56e1d6c1069c43f386a1359579b4c6387656bd5cc68b1394` |
| `runtime/launch.h` | 9285 | `e42df7b2a92d731b9f49e6ed0a09687ec563ef5893c5208b5d138fea49447078` |
| `runtime/pack/fp8_resident_pack.py` | 20275 | `f86bca7aeafd4d635528c4cd3fdd3db92312cb751452b3b9a73da6e8699f3946` |
| `runtime/pack/stage_pack.py` | 17823 | `5f9a57e020461b198658b357cca4fe6ffa1e232f34547de58c8b2b19c9cdfb83` |
| `runtime/tensor_map.h` | 4149 | `a1496d86729536135c0ccd283442ac6d2cc56c45e1a20ca5675ca98cc6a577c5` |
| `scheduler/long_context.c` | 24140 | `6187df321629a6697b8b91c0c921809d3f756b5ea12933e4a30e7780ec36240c` |
| `tests/host_cuda/k3_slice_host.cu` | 14500 | `89af1ba1c316a51a127bed86f0c4e2bb2b030fa58a4459a53c72931e1bc37bc2` |
| `tests/host_cuda/shim/inference/kernels/gemm.cuh` | 480 | `9e60fb80589510db5fc6f2ca6fba85f76ffab03c1cdd594c64f25af029e8fd9f` |
| `tests/host_cuda/shim/runtime/gemm.cuh` | 4740 | `b7cd06ccbd1ed7fe6b88b92bdb428e6dad19880e539fe49ac7a47bbf4858a3fa` |
| `tests/studies/sparkpipe_glm52_pipesim.c` | 29832 | `aeaa8ccc170cf300c6a74be70a36145fd4755cdd6c2ed595e10cf1edd869b6d9` |
| `tests/test_code_size.py` | 1677 | `544e96894d20e73eb15d39f7975ecbf2bc30e780228637d828caaa61df494003` |
| `tests/test_dry_law.py` | 6912 | `8b4d1a8c4aadd84611f0bda430ae8b644924973b99e75f8c6eb233c29d1f2586` |
| `tests/test_glm52_fp8_pack_layout.py` | 3688 | `e4dba2365c236d49464ffed1c5b384c95c4e0bb6f0b7828a8e8eaf564fae4588` |
| `tests/test_glm52_http_gateway.c` | 11430 | `17faf27247c29878802b896cb3a25e3bffd98ba37baeceec5dfef3818ced95fe` |
| `tests/test_glm52_kv_cache.c` | 40771 | `92d79c0f4c86318e00486ca5d195c9be8dbeb017aba7f991a24142286e4b9b72` |
| `tests/test_glm52_quantized_cuda_contract.py` | 5547 | `b2f0dba0daf1ef07a862b2132014b85131c7ec66c08dc2b8fb892470383797a7` |
| `tests/test_glm52_resident_decode_stage_firmware.c` | 180676 | `d5bb9858a8ba7c841e47ac2c3322f8fd5a795771c076d03f337328753bd4d059` |
| `tests/test_glm52_ring_rank_daemon.c` | 10303 | `8854c0b1802820f9fe909cd30fdace12d40532ebcd0f273adf252e63c45f75de` |
| `tests/test_glm52_ring_runtime.c` | 14071 | `9784f9eeaee9fba923c5bcd22c0e6b69aa900bfd4da58011b57306b118e3cfd4` |
| `tests/test_glm52_scheduler.c` | 50268 | `3dd25953679d97befc4f3fd6c7d4026823b653ef3b2a9788bb10d0cc4b340a58` |
| `tests/test_glm52_serving_engine.c` | 46826 | `ea8b14f1226f7b0e3122ca4348918855d09b19e5fc855a5845cf8374756343de` |
| `tests/test_glm52_stage_pack.py` | 8203 | `581e2be016c40b8ca937afe424e3f3db58a363398406b893839edb193fe6d235` |
| `tests/test_k3_quant_recipe.py` | 5414 | `6e3ec06d22c3e99113f92f2681adbe52f774dcc7bc1a038ebcdde2a87a39ac09` |
| `tests/test_launch.c` | 4501 | `f7c960d254324e92f1e1ab00a1dc37f03ce4a12f278a0ca63ece0deedfff3adf` |
| `tests/test_layer_kinds.py` | 8963 | `6cb9ff5b6daa9fd7e2a80658cd893da5548af72b412c096d0450436a8aa8f8aa` |
| `tests/test_measured_status.py` | 2695 | `475140286ad27dc2f9f2ad882f260d4fa592836e7d4af6638eee2a3f7a00931d` |
| `tests/test_memory_contracts.py` | 17490 | `5beefade797f05d21f0a6a3dd11bb3527dbc4dc2eefb8972fd3bc07e73cbaea0` |
| `tests/test_model_description.c` | 7708 | `84a793804ef77a3412f702948284b0087f894e3ad59261624996c31fad8b63b0` |
| `tests/test_model_driver_contracts.py` | 3101 | `8f5dcef38962a0ed9721860e67bbcefb574fd946ac1ac95564b7b1965e6777a8` |
| `tests/test_release_assemble.py` | 11686 | `96117200a6d8a391def2272a17411c27cf384f95cbb3a08bb9dccbc6a5583f54` |
| `tests/test_router_precision_contract.py` | 1565 | `abe0374ff81576848927f0f92dbd05146a22929a358f2d883acc47ea8563332e` |
| `tests/test_tensor_map_encode.c` | 2299 | `01df22232760e915a1501f7e91d00dfff4ec9a94b741e33cb50eaae43786c116` |
| `tests/test_tensor_map_geometry.c` | 3313 | `8d5400ce8c2b72ca6211fc9560db7b06f4aa6c129e35a96af5d873c8ce670746` |
| `tools/cuda13_sm121a_compile_gate.sh` | 6301 | `fc372bcefa09419dddfca7d5f797491fab3b4d12c7cd331aabf80cc51e6a8bbe` |
| `tools/gates.sh` | 10057 | `94e71e48969623f0946cff9f44700465233ff757e63b1911daeed98bdcf7923a` |
| `tools/get_cuda.sh` | 2352 | `e044cd324484e9a8435617697d1b1db94a1ec6346a9e0478806f41eea8298112` |
| `tools/glm52_model_contract.py` | 12338 | `9fac72dc29ef961b9a5807d08c91765caec31d260618cebbafd20d7503a89638` |
| `tools/glm52_resident_pack_common.py` | 3697 | `89d4cc7dbd580d62891e8747b972ac130e7fa189bca5293dc2791b9f492b1bcc` |
| `tools/sparkpipe_release_assemble.py` | 14642 | `6080969b60160a2471e9ccec80f89f13715eac7d077e33233942abeae4fb9b42` |

## Deleted from Proposed Tree

| Path | Bytes | SHA-256 |
|---|---:|---|
| `docs/validation-logs/clean_build.log` | 120 | `5bbf1ddbdd715970739b52586ae44e1c4bb850037316a0d184f620d565201aef` |
| `docs/validation-logs/core_boundary_audit.log` | 331 | `60bddefc63955436a687471616bbd1bb6308a153417951c06d7aa20ec7936202` |
| `docs/validation-logs/cuda_node_context_builder.log` | 40 | `83b3dbade4fdf2607495f25dde281fbce80bdb16ef49c26a4442c6102ae0411c` |
| `docs/validation-logs/host_build.log` | 25001 | `af169cb8b33088213660e6806a60c25daf116f4a02dc85218d3a8c3cc841f4cb` |
| `docs/validation-logs/host_test_suite.log` | 24052 | `d8261c5e88b34860fd30cd1b93638bac1ade2d6be28057ffbece049047306243` |
| `docs/validation-logs/host_test_suite_without_loopback.log` | 9772 | `848c7bbd9539feeb8978400e971fa11001629f1b524a13b80f305c066f36b8d6` |
| `docs/validation-logs/memory_contracts.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `docs/validation-logs/non_glm_model_driver_contracts.log` | 897 | `e34ade0f62eaf1b6b9eb5b25016f7c8f36fe5f256452b41d08cf9b9d3ebdda0c` |
| `docs/validation-logs/python_tool_syntax.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `docs/validation-logs/required_host_targets.log` | 97 | `f302765dc2ebcdaf901435c43e5213a6bde2966eaeefa8286ce1212223efe5a1` |
| `docs/validation-logs/source_tree_stability.log` | 178 | `c6a9b743611729e9e8859a5b768bfe5715d50d68616352d2730cc93a394fe2d8` |
| `qualification/phase3_full_make_test_failure.log` | 6467 | `b86d17d0db10c3c206505043a50c39929ac1b5400399b42e32a4f8d7d7e1066c` |
| `qualification/phase3_targeted_validation.log` | 23684 | `f0392de4ddd7035d53c1e23bc0c96a9ef06baac4769ab9e550ce88521ab33a97` |
| `tests/test_glm52_cuda_resident_gate.c` | 4084 | `6bca9ac2372a1cd46b609a84465335942a8856e4c4590f1fab6c0d5f2f588976` |
