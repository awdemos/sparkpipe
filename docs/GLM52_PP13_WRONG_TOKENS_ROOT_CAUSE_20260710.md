# GLM52 PP13 wrong tokens — root cause for dev handoff

**Date:** 2026-07-10  
**Repo:** `sparkpipe` (`/Users/mac/sparkpipe`)  
**Symptom:** 13× SparkPipe ring inference mostly works (prefill/decode complete, tokens stream) but **generated tokens are wrong** (garbage text; often id `0` / `!`; sometimes nondeterministic at temp=0).

This note is a code + diagnostics verdict, not a “collect more data” plan.

---

## TL;DR for the assignee

| Rank | Bug | Status in tree | Effect |
|------|-----|----------------|--------|
| **1** | RoPE **interleaved** pairing vs trained **`rotate_half`** | Fixed in `b5bc7a8` / PR #235 | Token0 OK, **token1+ wrong**; ring still “serves” |
| **2** | Dense FP8 **prepared MLP requires `algorithm` backend that bind never sets** | **Still broken** (PR #247 incomplete) | Production dense path never runs as designed |
| 3 | Prefill race, pos=0 metadata, absorbed attention default, restricted-vocab greedy | Fixed in earlier PRs | Compound wrong / nondeterministic tokens on older releases |

**If the live ring is pre-RoPE-fix:** redeploy post-`b5bc7a8` first. That alone explains “works but wrong tokens.”  
**If post-RoPE and still wrong:** fix dense MLP bind (#2); do not keep silent fallbacks.

---

## Symptom evidence (in-repo)

### Matched prompt

```text
Say OK. OK.
tokens: 45494 10397 13 10397 13
```

### Diagnostics locations

```text
diagnostics/glm52_pp13_diff_20260708/
diagnostics/glm52_pp13_diff_20260709/
diagnostics/glm52_pp13_diff_20260709_release231_seq1/numeric_diff.txt
docs/GLM52_FP8_PP13_ATTENTION_MODE_ROOT_CAUSE_20260709.md
docs/GLM52_KV_SLOT_PROBE_20260709.md
```

### Numeric shape (release231 / post-tiled attention)

From `diagnostics/glm52_pp13_diff_20260709_release231_seq1/numeric_diff.txt`:

- **token0 rank0 vs oracle:** `rel_l2 ≈ 0.004`, cos ≈ 0.99999 (good)
- **token1+ rank0:** `JUMP`, cos collapses (bad)
- Corruption amplifies across ranks

Gateway often returned `id0=0` (`!`) even after tiled-attention work.

### Eliminated causes (documented)

From `docs/GLM52_FP8_PP13_ATTENTION_MODE_ROOT_CAUSE_20260709.md`:

- Hidden transport hop integrity OK  
- FP8 pack non-MoE bytes / MoE parser (when attention tiled)  
- Graph replay / builtin PP13 unroll  
- Runtime KV tables / physical block remap / MTP frame context  

From `docs/GLM52_KV_SLOT_PROBE_20260709.md`:

- KV **slot allocation** stable across warm runs  
- Wrong tokens still varied (`woman` / `ilib` / `9`) while slots stable  

---

## Bug #1 — RoPE pairing (primary wrong-token root cause)

### What was wrong

Production applied RoPE to **adjacent** elements `(2i, 2i+1)`.

GLM / HF default `rope_type` uses **`rotate_half`**: pair element `i` with `i + rope_dim/2`.

### Why it matches the failure perfectly

| Observation | Why RoPE pairing explains it |
|-------------|------------------------------|
| Pipeline completes | Attention still runs; math is self-consistent |
| Token **0** matches oracle | Position 0 → θ=0 → rotation is **identity** under either pairing |
| Token **1+** jumps hard | Position > 0 → Q/K rope wrong vs trained weights |
| key_nope / value can match while rope pair0 does not | Discriminator is specifically rope |
| Local CUDA validation passed | Reference mirrored the **same interleaved** bug |

### Fix in tree

```text
commit b5bc7a8  RoPE half-split pairing to match the trained weight convention
PR #235
```

Touch points in `modules/glm52_resident_decode_stage/source/spark_glm52_sm121_required_decode_stage.cu`:

- `PrepareKernel` — query rope branch  
- `PrepareKernel` — MLA cache key rope append  
- `DsaKeyNormRopeStoreKernel` — DSA key rope  

After fix, pair loads look like:

```c
input_offset = row * ROPE_DIMENSION + rope_pair_index;
ApplyRopePair(
    input[input_offset],
    input[input_offset + rope_pair_count],  // half-split, not +1
    cos, sin,
    &out[input_offset],
    &out[input_offset + rope_pair_count]);
```

### Cos/sin tables

Builder still indexes `inv_freq` by pair (`pair * 2 / rope_dim`) in  
`SparkGlm52Pp13BuilderInitializeTables` — that matches half-split pair index `i` with frequency for dimension `2i`, which is the standard table layout for both conventions; only the **element pairing** was wrong.

### Dev action

1. Confirm live release includes `b5bc7a8` (or later) on **all 13 ranks**.  
2. Re-run matched prompt + numeric diff: token1+ should collapse to precision-level vs serialized/oracle.  
3. Keep validation reference on `rotate_half` so self-consistent-wrong cannot pass again.

---

## Bug #2 — Dense FP8 prepared MLP cannot run on PP13 (still live)

### Builder declares PREBOUND quantized dense MLP

`modules/glm52_resident_decode_stage/source/spark_glm52_pp13_node_context_builder_cuda.cu`:

```c
// layers 0..FIRST_ROUTED_LAYER-1 (FIRST_ROUTED_LAYER == 3)
mlp_execution_mode =
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_QUANTIZED_TENSOR_CORE;
```

Dense weights wired as **FP8 only** (`dense_*_weight_fp8`).  
`dense_*_weight_bf16` is **never set** on the PP13 node context.

### Plans created without scaled-GEMM backend

`CreateQuantizedFp8One` → `plan_kind = TENSOR_CORE_FP8_E4M3_ROW_MAJOR`, workspace allocated, **`algorithm` left null**.

### Bind does not install `algorithm`

`SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedTensorCoreLinearPlan`:

```c
linear_plan->custom_launch_function =
    (void *)SparkGlm52Sm121RequiredDecodeStageLaunchBlackwellQuantizedTensorCoreLinearPlan;
// algorithm NOT set
return SPARK_STATUS_OK;
```

`BindFp8E4m3LinearScaledGemmBackend` exists but is **not called** from the PP13 builder.

### Prepared dense path requires `algorithm`

`LaunchFp8PreparedActivationWeightLinearPlan`:

```c
backend = (const Fp8ScaledGemmBackend *)linear_plan->algorithm;
if (backend == 0)
    return SPARK_STATUS_INVALID_ARGUMENT;
```

So **`TryLaunchFp8DenseMlpPreparedStaging` cannot succeed** on the PP13 production bind as written.

### Historical behavior (silent fallback)

Older code: `algorithm == 0` → `NOT_FOUND` → `LaunchPreboundDenseMlp` fell into `LaunchLinear` → prebound custom launch → built-in FP8 WMMA path (ignores null BF16 weight pointers). Numerically usable, but not the “prepared FP8 dense” path probes instrumented.

PR #246 already concluded prepared path returned not-found for layer2; compute went through the LaunchLinear/WMMA fallback.

### PR #247 (2026-07-10) — incomplete

```text
42e6b32 Fix FP8 prepared dense MLP plan gate
PR #247
```

- Removed early `algorithm == 0 → NOT_FOUND`  
- If prepared fails and mode is PREBOUND: **`INVALID_ARGUMENT`** (no WMMA fallback)

That **does not bind a backend**. It fail-closes (or forces a mid-path `INVALID_ARGUMENT` from prepared launch) without fixing the missing bind. Incomplete for production.

### Dev action (pick one, do not leave silent wrongness)

**Option A (minimal, honest):**  
If no scaled-GEMM backend is bound, skip prepared staging and use the existing plan launch (`LaunchLinear` / built-in FP8 WMMA) **explicitly**, and assert plans have `custom_launch_function`. Fail only if that also fails.

**Option B (production intended path):**  
In PP13 builder after plan create, bind a real `Fp8ScaledGemmBackend` into dense gate/up/down via `BindFp8E4m3LinearScaledGemmBackend`, then keep fail-closed PREBOUND.

**Option C:**  
Make `LaunchFp8PreparedActivationWeightLinearPlan` fall through to the same built-in WMMA path `LaunchBlackwellBuiltInQuantizedTensorCoreLinearPlan` uses when `algorithm == 0` (after activation quantize), and only require backend when one is installed.

Do **not** leave PREBOUND mode taking a path that reads null BF16 dense weights without a plan.

---

## Already-fixed companions (deploy checklist)

| Issue | Commit / PR | Notes |
|-------|-------------|--------|
| Absorbed-latent attention default | PR #222 / tiled default | Absorbed path corrupt vs oracle; keep tiled online softmax |
| Serial prefill race (shared buffers) | `1b1bf45` / PR #240 | Drain runner idle between tokens |
| Mid-ring submit drain | `c03f80d` / PR #241 | |
| Serial prefill position always 0 | `417c6a7` / PR #230 | Broke pos>0 RoPE even with correct pairing |
| Full-vocab greedy epilogue (no MTP) | `bf94fc6` / PR #238 | Restricted 256-vocab greedy → junk tokens |
| Serialized KV addressing / metadata | PRs #226–#233 | Plumbing vs oracle |

Older release without this set will still emit wrong tokens even if “the ring is up.”

---

## Related docs

```text
docs/GLM52_FP8_PP13_ATTENTION_MODE_ROOT_CAUSE_20260709.md
docs/GLM52_KV_SLOT_PROBE_20260709.md
docs/GLM52_ABSORBED_MLA_DECODE_20260704.md
docs/CODEX_RUNBOOK.md
```

Key source files:

```text
modules/glm52_resident_decode_stage/source/spark_glm52_sm121_required_decode_stage.cu
  - ApplyRopePair / PrepareKernel / tiled attention / dense MLP / final epilogue
modules/glm52_resident_decode_stage/source/spark_glm52_pp13_node_context_builder_cuda.cu
  - wire layer, bind plans, cos/sin tables, serial prefill
modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_linear_plan.cu
  - CreateQuantizedFp8One (algorithm left null)
```

---

## Suggested message to send a teammate

> PP13 wrong-token writeup is in-repo at  
> `docs/GLM52_PP13_WRONG_TOKENS_ROOT_CAUSE_20260710.md`.  
>  
> Primary: RoPE was interleaved; weights need rotate_half — fixed in `b5bc7a8`, must be on the live ring.  
> Still open: dense FP8 prepared MLP requires `plan->algorithm` scaled-GEMM backend but bind never sets it; PR #247 fail-closes without binding. See section “Bug #2”.

---

## How to share this file

**Local path (this machine):**

```text
/Users/mac/sparkpipe/docs/GLM52_PP13_WRONG_TOKENS_ROOT_CAUSE_20260710.md
```

**Relative path in repo:**

```text
docs/GLM52_PP13_WRONG_TOKENS_ROOT_CAUSE_20260710.md
```

**To a remote dev:** commit/push this file, or attach/email the markdown, or paste:

```text
See docs/GLM52_PP13_WRONG_TOKENS_ROOT_CAUSE_20260710.md on main (or attach that file).
```
