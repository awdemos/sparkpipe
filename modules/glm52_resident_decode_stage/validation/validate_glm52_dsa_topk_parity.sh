#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 MODULE_ARCHIVE" >&2
    exit 2
fi

module_archive="$1"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_directory="$(cd "${script_directory}/.." && pwd)"
repository_root="$(cd "${module_directory}/../.." && pwd)"
validation_directory="$(mktemp -d)"
trap 'rm -rf "${validation_directory}"' EXIT
common_target="build/libsparkpipe_common.a"
runtime_target="build/libsparkpipe_runtime.a"
compiler_target="build/libsparkpipe_compiler.a"
common_archive="${repository_root}/build/libsparkpipe_common.a"
runtime_archive="${repository_root}/build/libsparkpipe_runtime.a"
compiler_archive="${repository_root}/build/libsparkpipe_compiler.a"
nvcc_path="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121a}"

if [[ ! -s "${module_archive}" ]]; then
    echo "module archive is missing or empty" >&2
    exit 2
fi
if ! command -v "${nvcc_path}" >/dev/null 2>&1; then
    echo "nvcc unavailable for DSA top-k parity validation" >&2
    exit 2
fi
if [[ "${cuda_architecture}" != "sm_121a" ]]; then
    echo "DSA top-k parity admits only sm_121a required-CUDA artifacts" >&2
    exit 2
fi

required_cuda_link_args=()
if [[ -n "${GLM52_REQUIRED_CUDA_LINK_ARGS:-}" ]]; then
    read -r -a required_cuda_link_args <<< "${GLM52_REQUIRED_CUDA_LINK_ARGS}"
fi
required_cuda_library_path=""
for required_cuda_link_arg in "${required_cuda_link_args[@]}"; do
    if [[ "${required_cuda_link_arg}" == *.so ]]; then
        required_cuda_library_directory="$(cd "$(dirname "${required_cuda_link_arg}")" && pwd)"
        if [[ -z "${required_cuda_library_path}" ]]; then
            required_cuda_library_path="${required_cuda_library_directory}"
        else
            required_cuda_library_path="${required_cuda_library_path}:${required_cuda_library_directory}"
        fi
    fi
done
if [[ -n "${required_cuda_library_path}" ]]; then
    export LD_LIBRARY_PATH="${required_cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

make -C "${repository_root}" "${common_target}" "${runtime_target}" "${compiler_target}"

"${nvcc_path}" \
    -std=c++17 \
    -O3 \
    --use_fast_math \
    -arch="${cuda_architecture}" \
    -I"${repository_root}/include" \
    -I"${module_directory}/include" \
    -I"${module_directory}/source" \
    "${script_directory}/spark_glm52_dsa_topk_parity.cu" \
    "${module_archive}" \
    "${runtime_archive}" \
    "${compiler_archive}" \
    "${common_archive}" \
    "${required_cuda_link_args[@]}" \
    -lcublasLt \
    -lcublas \
    -ldl \
    -o "${validation_directory}/glm52_dsa_topk_parity"

"${validation_directory}/glm52_dsa_topk_parity"
