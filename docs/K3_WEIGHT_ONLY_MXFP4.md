# K3 is weight-only MXFP4

## What the checkpoint says

`quantization_config` quantises `Linear` weights at 4 bits, group 32, symmetric,
`scale_dtype` `uint8` — and sets **`input_activations: null`** and
`output_activations: null`. The ignore list excludes attention, shared experts,
the dense MLP, `lm_head` and the vision tower.

The tech report's deployment section says activations were computed in MXFP8.
That describes Moonshot's own serving stack. It is not what the released
checkpoint asks an inference engine to do, and the two were conflated in
`config.h` until a second audit separated them.

## What this tree does instead

`K3Quantise<Format>` quantises the expert activations to **MXFP4** — not even
the MXFP8 the report mentions — because `LmGemmLaunch` takes one `Format` for
both operands. Three call sites: the dense projection helper and both expert
GEMMs.

The weight decoder is right. The ABI around it is wrong.

## The correct data path

    BF16 activation
      x streamed MXFP4 weight
      -> decode the weight tile to BF16 registers
      -> BF16 MMA, FP32 accumulation
      -> BF16 output

## What it takes

1. **An asymmetric GEMM.** `LmGemmLaunchAsymmetric<LmBf16Format, LmMxfp4>`:
   A tile at 16 stored bits with no scale, B tile at 4 bits with an E8M0 scale
   every 32, the existing BF16 MMA and FP32 accumulators. The symmetric launcher
   stays as an explicit same-format wrapper for the other models.

2. **No activation quantiser on the expert path.** The first expert GEMM needs
   the route expansion the quantiser was doing implicitly, as a BF16 gather:
   `latent_bf16 + route_source_token -> expert-major`. The second activation is
   already expert-major after SiTU.

3. **Decode E8M0 in the weight load.** `LmE8m0ToFloat` exists and is unused;
   the GEMM takes `const float *scale_b`. Pre-expanding scales to FP32 costs
   17,547,264 -> 20,643,840 bytes per expert layer, 17.6% of expert bandwidth,
   for a value that decodes in one instruction.

4. **One scale layout, asserted.** The quantiser writes row-major and the GEMM
   reads as though K-group-major. `[expert][output_neuron][k_group]` with the
   packer and kernel sharing the contract.

5. **A packer.** `bind.cu` says no K3 pack format exists. It must preserve the
   MXFP4 payload and U8 scales, concatenate `w1` and `w3` into the combined
   gate/up layout, keep `w2` orientation, validate 896 experts and group 32,
   reject E8M0 `0xff`, and fail loudly on a recipe mismatch.

## Validation the gate does not yet do

A static assertion that the expert GEMMs are `A=BF16, B=MXFP4`; MXFP4+E8M0
decode against a dequantised BF16 reference; route-gather bounds; a full
`w1/w3 -> SiTU -> w2` numeric comparison; exact packed-byte accounting; and an
SM121 build with a GPU comparison, which nothing here can do.

## Why the gate is red rather than softened

`tests/test_k3_quant_recipe.py` passed while this was wrong, because it only
asked which *weight* projections took the quantised `Format`. It now also asks
whether any activation does, and fails. Leaving it failing is deliberate: a
suite that passes over a known-wrong data path is worth less than one that says
where it is wrong.
