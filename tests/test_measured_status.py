#!/usr/bin/env python3

import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT / "examples" / "model_descriptions" /
    "glm52_resident_decode_stage_firmware.json"
)
ACTIVE_CUDA_DIRECTORIES = (
    ROOT / "inference" / "llms" / "glm5_2",
    ROOT / "modules" / "glm52_resident_decode_stage",
)
FORBIDDEN_FLAGS = {
    "jit_kv_cache",
    "no_host_staging",
    "no_device_memcpy",
    "driver_private_expert_queues",
    "bulk_prefill",
    "validated_latency",
}
POSITIVE_LATENCY = re.compile(
    r"(?:validated_maximum_latency_ns|validated_stage_latency_ns|"
    r"estimated_stage_latency_ns)\s*=\s*[1-9][0-9]*"
    r"(?:u|ul|ull)?\b",
    re.I,
)


def main():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    program = model["stages"][0]["programs"][0]
    scheduling = program["scheduling"]
    flags = set(scheduling["flags"])
    assert program["max_inflight"] == 1
    # Capacity is a build contract, not a performance claim. The health/status
    # surfaces below must continue to report the B1024 path as unmeasured until
    # a retained full-ring receipt exists.
    assert scheduling["max_active_slots"] == 1024 * 8
    assert scheduling["max_resident_sequences"] == 16384
    assert scheduling["validated_latency_ns"] == 0
    assert scheduling["private_queue_count"] == 1
    assert flags.isdisjoint(FORBIDDEN_FLAGS)
    assert "current_spark_topology" not in model["metadata"]

    health_source = (ROOT / "api" / "http_gateway.c").read_text(encoding="utf-8")
    backend_source = (ROOT / "node" / "backend.c").read_text(encoding="utf-8")
    assert "production_contract_flags" not in health_source
    assert "ring_control_ready" not in health_source
    assert "PRODUCTION_REQUIRED_FLAGS" not in backend_source
    assert '\\"accuracy_status\\":\\"NOT_MEASURED\\"' in health_source
    assert '\\"performance_status\\":\\"NOT_MEASURED\\"' in health_source

    cuda_sources = []
    for source_directory in ACTIVE_CUDA_DIRECTORIES:
        cuda_sources.extend(sorted(source_directory.rglob("*.cu")))
        cuda_sources.extend(sorted(source_directory.rglob("*.cuh")))
    assert cuda_sources, "no GLM-5.2 CUDA sources found"
    for source_path in cuda_sources:
        source = source_path.read_text(encoding="utf-8")
        assert POSITIVE_LATENCY.search(source) is None, source_path

    status = (
        ROOT / "docs" / "GLM52_MEASURED_STATUS.md"
    ).read_text(encoding="utf-8")
    for required in ("MEASURED", "OBSERVED", "NOT_MEASURED", "NOT_WORKING"):
        assert required in status
    assert "3.112" in status
    assert "2.884" in status


if __name__ == "__main__":
    main()
