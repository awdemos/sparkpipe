#!/usr/bin/env python3
"""DSv4 driver source contracts: the audit numbers must match the code.

The layer driver's header comment carries exact attention weight bytes and the
Pro launch budget is written into deepseek_v4_pro/unity.cu. Both are claims
about config geometry, and claims drift: this gate recomputes every figure
from inference/llms/deepseek_v4/config.h and model_contracts/dsv4_pro.json and
requires the recomputed decimal to appear in the comment, so editing a config
without re-auditing the comment fails here.

It also pins the two structural facts the 2026-08-01 audit established:

  - the KV latent GEMM reuses the low-rank path's quantised input instead of
    re-quantising the normed rows (one launch and one full hidden re-read per
    layer per token), guarded by an explicit input_dimension check;
  - the per-function launch counts, because launch tax is 18-49% of the
    50 tok/s Pro budget and every added launch is a regression.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"missing {label}: {needle}")


def define(text: str, name: str) -> int:
    match = re.search(rf"#define {name} (\d+)u\b", text)
    if not match:
        raise SystemExit(f"missing #define {name}")
    return int(match.group(1))


def function_body(text: str, name: str) -> str:
    start = text.index(f"static int32_t {name}(")
    brace = text.index("{", start)
    depth, index = 1, brace + 1
    while depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    return text[brace:index]


def main() -> None:
    layer = read("inference/llms/deepseek_v4/layer.cuh")
    flat = re.sub(r"\s+", " ", layer)
    config = read("inference/llms/deepseek_v4/config.h")
    pro = json.loads(read("model_contracts/dsv4_pro.json"))

    # -- exact attention weight bytes, recomputed from geometry ---------------
    hidden = define(config, "DSV4_HIDDEN")
    heads = define(config, "DSV4_ATTN_HEADS")
    head_dim = define(config, "DSV4_HEAD_DIM")
    rope_dim = define(config, "DSV4_ROPE_DIM")
    query_rank = define(config, "DSV4_QUERY_LORA_RANK")
    flash_per_layer = (
        hidden * query_rank
        + query_rank * heads * head_dim
        + hidden * (head_dim + rope_dim)
        + heads * head_dim * hidden
    )
    require(layer, str(flash_per_layer), "Flash per-layer attention weights")

    pm = pro["model"]
    p_hidden = pm["hidden_dimension"]
    p_q_dim = pm["attention_head_count"] * pm["head_dimension"]
    p_coded = (
        p_hidden * pm["query_lora_rank"]
        + pm["query_lora_rank"] * p_q_dim
        + p_hidden * (pm["head_dimension"] + pm["qk_rope_head_dimension"])
        + p_q_dim * p_hidden
    )
    require(layer, str(p_coded), "Pro as-coded per-layer attention weights")
    groups = pm["output_group_count"]
    rank = pm["output_lora_rank"]
    o_lowrank = groups * ((p_q_dim // groups) * rank + rank * p_hidden)
    require(layer, str(o_lowrank), "Pro contract grouped low-rank o_proj")
    require(
        layer,
        str(p_coded - p_q_dim * p_hidden + o_lowrank),
        "Pro contract per-layer attention total",
    )

    # -- the KV latent GEMM reuses the low-rank quantised input ----------------
    require(
        flat,
        "b->query_scratch.input_codes,b->kv_latent_weight",
        "KV latent GEMM activation reuse",
    )
    require(
        flat,
        "b->query.input_dimension != DSV4_HIDDEN",
        "reuse contract guard",
    )
    attention = function_body(layer, "Dsv4LayerAttention")
    quantises = attention.count("LM_LAUNCH((LmQuantiseRowsKernel")
    if quantises != 1:
        raise SystemExit(
            f"attention path runs {quantises} quantise launches, expected 1 "
            "(the normed-rows quantise rides the low-rank path)"
        )

    # -- launch-count budget, the 50 tok/s Pro lever ---------------------------
    moe = function_body(layer, "Dsv4LayerMoe")
    a_launches = attention.count("LM_LAUNCH(")
    a_gemms = len(re.findall(r"LmGemmLaunch(?:Asymmetric)?<", attention))
    m_launches = moe.count("LM_LAUNCH(")
    m_gemms = len(re.findall(r"LmGemmLaunch(?:Asymmetric)?<", moe))
    m_routes = moe.count("LmRouteBuild<")
    # LmLowRankProject expands to five launches: quantise, GEMM, norm,
    # quantise, GEMM. Three attention LM_LAUNCH sites sit inside the sparse
    # branch. Dense layer = (a_launches - 3) + a_gemms + 5 + m_*; sparse adds 3.
    dense = (a_launches - 3) + a_gemms + 5 + m_launches + m_gemms + m_routes
    sparse = dense + 3
    if (dense, sparse) != (29, 32):
        raise SystemExit(
            f"layer launch budget moved: dense {dense}, sparse {sparse} "
            "(audited at 29/32 on 2026-08-01; update deepseek_v4_pro/unity.cu "
            "and this gate together if the change is deliberate)"
        )

    # -- audit flags that must not be silently dropped -------------------------
    require(layer, "65,535", "sparse-score grid.y limit note")
    require(layer, "key graphs on both", "graph-capture shape key note")
    require(layer, "WINDOW:", "sparse window-clamp semantics flag")
    require(layer, "SPAN:", "query rope per-head span flag")
    require(layer, "WIDTH:", "query row-width flag")
    pro_unity = read("inference/llms/deepseek_v4_pro/unity.cu")
    require(pro_unity, "LAUNCH BUDGET", "Pro launch-budget audit note")

    print("PASS DSv4 driver source contracts")


if __name__ == "__main__":
    main()
