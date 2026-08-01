#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"missing {label}: {needle}")


def reject(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise SystemExit(f"forbidden {label}: {needle}")


def main() -> None:
    route = read("inference/kernels/route.cuh")
    require(route, "packed_rows != expected_packed_rows", "route cardinality validation")
    require(route, "LmLaunchGroupedTileM(rows,top_k,EXPERTS)", "token-priced grouped tile")

    contracts = {
        "k3": (
            "inference/llms/kimi_k3/layer.cuh",
            "LM_TOPK_SCORE_SIGMOID",
            "K3_ROUTED_SCALE",
        ),
        "glm52": (
            "inference/llms/glm5_2/layer.cuh",
            "LM_TOPK_SCORE_SIGMOID",
            "GLM52_ROUTED_SCALE",
        ),
        "dsv4": (
            "inference/llms/deepseek_v4/layer.cuh",
            "LM_TOPK_SCORE_SQRT_SOFTPLUS",
            "DSV4_ROUTED_SCALE",
        ),
        "mimo25": (
            "inference/llms/mimo_2_5/layer.cuh",
            "LM_TOPK_SCORE_IDENTITY",
            "1.0f",
        ),
    }
    for family, (path, transform, scale) in contracts.items():
        text = read(path)
        require(text, "LmRouteBuild<", f"{family} device route build")
        require(text, "group_tile_prefix_w1", f"{family} W1 prefix")
        require(text, "group_tile_prefix_w2", f"{family} W2 prefix")
        require(text, "prefix_built = 1u", f"{family} prebuilt prefix")
        require(text, transform, f"{family} router transform")
        require(text, scale, f"{family} router scale")
        reject(text, "LmLaunchGroupedTileM(packed_rows", f"{family} route-priced tile")

    k3 = read("inference/llms/kimi_k3/layer.cuh")
    require(k3, "b->expert_w1_weight,packed_rows,rows,", "K3 W1 token count")
    require(k3, "b->expert_w2_weight,packed_rows,rows,", "K3 W2 token count")

    dsv4 = read("inference/llms/deepseek_v4/layer.cuh")
    mimo = read("inference/llms/mimo_2_5/layer.cuh")
    require(dsv4, "float *router_logits;", "DSV4 FP32 router output")
    require(mimo, "float *router_logits;", "MiMo FP32 router output")
    reject(dsv4, "(uint32_t *)b->head_candidate_score", "DSV4 scratch alias")

    model = read("tests/studies/sparkpipe_glm52_batchplane_model.c")
    require(model, "expert_sweeps_per_active_expert = 1.0", "one expert sweep model")
    require(model, "replay/chunk expert-sweep multiplier: 1.0", "removed replay multiplier")
    reject(model, "BP_LAYERS * (BP_EXPERTS", "old queue-depth divisor")

    queue_header = read(
        "model-families/glm52/include/sparkpipe/spark_glm52_expert_queue.h"
    )
    queue_source = read(
        "model-families/glm52/src/spark_glm52_expert_queue.c"
    )
    require(queue_header, "SPARK_GLM52_EXPERT_QUEUE_MODE_SEALED_BATCH", "sealed mode")
    require(queue_header, "SparkGlm52ExpertQueueSealLayer", "seal API")
    require(queue_source, "queue->layer_sealed[layer_index]", "sealed layer state")

    print("PASS grouped-MoE source contracts")


if __name__ == "__main__":
    main()
