#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_directory="${SPARK_CUDA_GATE_OUTPUT_DIRECTORY:-${repository_root}/build/cuda13_sm121a_gate}"
nvcc_binary="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121a}"
gate_scope="${SPARK_CUDA_GATE_SCOPE:-complete}"

if [[ "${cuda_architecture}" != "sm_121a" ]]; then
    echo "CUDA gate requires CUDA_ARCH=sm_121a, got ${cuda_architecture}" >&2
    exit 2
fi
if ! command -v "${nvcc_binary}" >/dev/null 2>&1; then
    echo "CUDA gate requires nvcc from CUDA 13" >&2
    exit 2
fi
if [[ "${gate_scope}" != "existing" && "${gate_scope}" != "complete" ]]; then
    echo "SPARK_CUDA_GATE_SCOPE must be existing or complete" >&2
    exit 2
fi

nvcc_version="$(${nvcc_binary} --version)"
if ! grep -Eq 'release 13\.' <<<"${nvcc_version}"; then
    echo "CUDA gate requires CUDA 13.x" >&2
    printf '%s\n' "${nvcc_version}" >&2
    exit 2
fi

rm -rf "${output_directory}"
mkdir -p "${output_directory}/objects" "${output_directory}/ptx" "${output_directory}/logs"
printf '%s\n' "${nvcc_version}" > "${output_directory}/nvcc-version.txt"
printf 'CUDA_ARCH=%s\nSCOPE=%s\n' "${cuda_architecture}" "${gate_scope}" > "${output_directory}/configuration.txt"

cat > "${output_directory}/probe.cu" <<'PROBE'
#include <cuda_runtime.h>
#include <cstdint>

__global__ void SparkSm121aProbe(float *output, const float *input)
{
    uint32_t index;

    index = blockIdx.x * blockDim.x + threadIdx.x;
    output[index] = input[index] * 2.0f;
}
PROBE

"${nvcc_binary}" \
    -std=c++17 \
    -gencode arch=compute_121a,code=sm_121a \
    -lineinfo \
    -Xptxas=-v \
    -c "${output_directory}/probe.cu" \
    -o "${output_directory}/objects/probe.sm_121a.o" \
    2> "${output_directory}/logs/probe.ptxas.txt"

"${nvcc_binary}" \
    -std=c++17 \
    -arch=compute_121a \
    -ptx "${output_directory}/probe.cu" \
    -o "${output_directory}/ptx/probe.compute_121a.ptx"

common_flags=(
    -std=c++17
    -gencode
    arch=compute_121a,code=sm_121a
    --expt-relaxed-constexpr
    -lineinfo
    -Xptxas=-v
    -I"${repository_root}"
    -I"${repository_root}/include"
    -I"${repository_root}/model-families/glm52/include"
    -I"${repository_root}/model-families/qwen36/include"
    -I"${repository_root}/model-families/dsv4/include"
    -I"${repository_root}/model-families/k3/include"
    -I"${repository_root}/model-families/mimo25/include"
    -I"${repository_root}/modules/glm52_resident_decode_stage/include"
    -I"${repository_root}/modules/glm52_resident_decode_stage/source"
    -I"${repository_root}/modules/glm52_dspark_draft_backend/include"
)

translation_units=(
    tools/hardware/spark_cuda_characterize.cu
    tools/hardware/spark_nvme_characterize.cu
    inference/llms/deepseek_v4/unity.cu
    inference/llms/deepseek_v4_pro/unity.cu
    inference/llms/glm5_2/bind.cu
    inference/llms/glm5_2/unity.cu
    inference/llms/kimi_k3/bind.cu
    inference/llms/kimi_k3/unity.cu
    inference/llms/mimo_2_5/bind.cu
    inference/llms/mimo_2_5/unity.cu
    inference/llms/qwen_3_6/bind.cu
    inference/llms/qwen_3_6/unity.cu
    inference/stage/dispatch.cu
    inference/stage/serving_adapter.cu
    modules/glm52_dspark_draft_backend/source/spark_glm52_dspark_draft_backend.cu
    modules/glm52_dspark_draft_backend/validation/validate_glm52_dspark_epoch3_cuda.cu
    modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_fp8_moe_plan.cu
)

for relative_source in "${translation_units[@]}"; do
    source_path="${repository_root}/${relative_source}"
    if [[ ! -f "${source_path}" ]]; then
        echo "required CUDA translation unit missing: ${relative_source}" >&2
        exit 3
    fi
    artifact_name="${relative_source//\//__}"
    artifact_name="${artifact_name%.cu}"
    "${nvcc_binary}" \
        "${common_flags[@]}" \
        -c "${source_path}" \
        -o "${output_directory}/objects/${artifact_name}.o" \
        2> "${output_directory}/logs/${artifact_name}.ptxas.txt"
    "${nvcc_binary}" \
        -std=c++17 \
        -arch=compute_121a \
        --expt-relaxed-constexpr \
        -I"${repository_root}" \
        -I"${repository_root}/include" \
        -I"${repository_root}/model-families/glm52/include" \
        -I"${repository_root}/model-families/qwen36/include" \
        -I"${repository_root}/model-families/dsv4/include" \
        -I"${repository_root}/model-families/k3/include" \
        -I"${repository_root}/model-families/mimo25/include" \
        -I"${repository_root}/modules/glm52_resident_decode_stage/include" \
        -I"${repository_root}/modules/glm52_resident_decode_stage/source" \
        -I"${repository_root}/modules/glm52_dspark_draft_backend/include" \
        -ptx "${source_path}" \
        -o "${output_directory}/ptx/${artifact_name}.compute_121a.ptx"
    if ! grep -Eq '^\.target[[:space:]]+sm_121a' \
        "${output_directory}/ptx/${artifact_name}.compute_121a.ptx"; then
        echo "architecture-specific PTX target missing: ${relative_source}" >&2
        exit 4
    fi
done

if [[ -f /usr/include/infiniband/verbs.h ]]; then
    for direct_mode in 0 1; do
        "${nvcc_binary}" \
            "${common_flags[@]}" \
            -DSPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT="${direct_mode}" \
            -c "${repository_root}/ring/transport/rdma.cu" \
            -o "${output_directory}/objects/rdma_mode_${direct_mode}.o" \
            2> "${output_directory}/logs/rdma_mode_${direct_mode}.ptxas.txt"
    done
else
    echo "libibverbs headers unavailable; RDMA CUDA compile was not exercised" > \
        "${output_directory}/logs/rdma-skipped.txt"
fi

if [[ "${gate_scope}" == "complete" ]]; then
    make -C "${repository_root}/modules/glm52_resident_decode_stage" \
        clean archive \
        NVCC="${nvcc_binary}" \
        CUDA_ARCH=sm_121a \
        > "${output_directory}/logs/glm52-resident-stage-build.txt" 2>&1

    # The module archive never goes through the PTX check above; assert the
    # same sm_121a target on its objects so a wrong-arch module build (the
    # -arch flag drops the a suffix) fails the gate. Host-only objects carry
    # no cubin and are skipped.
    if ! command -v cuobjdump >/dev/null 2>&1; then
        echo "CUDA gate complete scope requires cuobjdump for the archive arch check" >&2
        exit 2
    fi
    module_build_directory="${repository_root}/build/modules/glm52_resident_decode_stage"
    for object_file in "${module_build_directory}"/*.o; do
        elf_listing="$(cuobjdump --list-elf "${object_file}" 2>/dev/null || true)"
        if [[ -n "${elf_listing}" ]] && ! grep -q 'sm_121a' <<<"${elf_listing}"; then
            echo "module archive object missing sm_121a target: ${object_file}" >&2
            exit 4
        fi
    done
fi

if command -v cuobjdump >/dev/null 2>&1; then
    for object_file in "${output_directory}"/objects/*.o; do
        object_name="$(basename "${object_file}")"
        cuobjdump --list-elf "${object_file}" > \
            "${output_directory}/logs/${object_name}.elf.txt" 2>&1 || true
        cuobjdump --dump-resource-usage "${object_file}" > \
            "${output_directory}/logs/${object_name}.resources.txt" 2>&1 || true
    done
fi

(
    cd "${output_directory}"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

echo "PASS CUDA 13 exact sm_121a compile gate (${gate_scope})"
