#pragma once

// DSpark, the speculative drafter for Kimi K3.
//
// Read from the released DSparkDraftModel config.json, 2026-07-27. Everything
// below is DISCLOSED.
//
// THE ARCHITECTURE WAS ALREADY HERE. spark_glm52_dspark.h models exactly this
// drafter for GLM-5.2: hidden taps at named target layers, a block size, a
// low-rank Markov bias, a confidence head in milli units, draft layer and head
// counts. Nothing structural is missing. What was missing is that every one of
// those constants resolves to a SPARK_GLM52_MODEL_* value, so pointing K3 at it
// would have run GLM-5.2's drafter shape against K3's target.
//
// Side by side, which is the argument for this file existing at all:
//
//                        GLM-5.2              Kimi K3
//   block size           8                    7
//   max verify tokens    7                    (block - 1, presumed)
//   aux layer ids        8,23,39,55,70        7,23,51,67,83
//   draft layers         5                    5
//   attention heads      64                   64
//   KV heads             64                   16     <- 4x fewer, GQA
//   head dimension       64                   64
//   intermediate         12288                14336
//   markov rank          256                  256
//   mask token           154856               163824
//
// The KV head count is the one that would not have announced itself: 64 versus
// 16 is a four-fold difference in the drafter's KV cache, and every other head
// number matches.

#include <stdint.h>

#include "inference/llms/kimi_k3/config.h"

#define K3_DSPARK_BLOCK_SIZE 7u
#define K3_DSPARK_DRAFT_LAYER_COUNT 5u
#define K3_DSPARK_AUX_LAYER_COUNT 5u
#define K3_DSPARK_AUX_LAYER_IDS_INITIALIZER { 7u, 23u, 51u, 67u, 83u }

// The drafter is a Qwen3-shaped transformer, not a copy of the target: five
// full-attention layers, GQA at 64 query heads over 16 KV heads, head dim 64.
#define K3_DSPARK_DRAFT_ATTENTION_HEAD_COUNT 64u
#define K3_DSPARK_DRAFT_KV_HEAD_COUNT 16u
#define K3_DSPARK_DRAFT_HEAD_DIMENSION 64u
#define K3_DSPARK_DRAFT_INTERMEDIATE_DIMENSION 14336u
#define K3_DSPARK_HIDDEN_DIMENSION K3_HIDDEN
#define K3_DSPARK_FULL_VOCAB_SIZE K3_VOCAB
#define K3_DSPARK_RMS_EPSILON 1e-05f

// THE DRAFTER ROTATES AND THE TARGET DOES NOT. K3 is NoPE throughout - the
// backbone calls no rope kernel and tests/test_rope_pairing.py enforces that.
// The drafter sets rope_theta 10000 with rope_type "default", so it carries
// ordinary rotary embedding over its own five layers.
//
// That asymmetry is legitimate: the drafter is a short-context Qwen3 model that
// reads taps from the target, not a small copy of it. But a gate that reasons
// "kimi_k3 is NoPE, therefore no rope kernel" would be wrong about this file,
// which is why the drafter's rope is declared here rather than left implicit.
#define K3_DSPARK_DRAFT_ROPE_THETA 10000.0f
#define K3_DSPARK_DRAFT_HEAD_ROPE_DIM K3_DSPARK_DRAFT_HEAD_DIMENSION

// The MASK token drives DFlash's noise stream. It is a real vocabulary id, not
// a sentinel, and it is 163824 - inside K3_VOCAB, near the special-token block
// that also holds bos 163584 and eos 163586.
#define K3_DSPARK_MASK_TOKEN_ID 163824u

// A low-rank learned bigram bias added to the draft logits, conditioned on the
// previous token: bias = W2(W1[token]), with W1 an embedding into rank 256 and
// W2 a bias-free linear back to the full vocabulary. It shapes the per-token
// distribution without touching the backbone.
#define K3_DSPARK_MARKOV_RANK 256u

// The confidence head predicts a per-position acceptance probability and drives
// adaptive block length. Its input is the hidden state CONCATENATED WITH THE
// MARKOV LATENT when confidence_head_with_markov is set, which it is - so the
// head is 7168 + 256 wide, not 7168. Sizing it at the hidden alone would build
// a head that runs and reads the wrong features.
#define K3_DSPARK_CONFIDENCE_WITH_MARKOV 1u
#define K3_DSPARK_CONFIDENCE_INPUT_DIMENSION \
	(K3_DSPARK_HIDDEN_DIMENSION + K3_DSPARK_MARKOV_RANK)

// Confidence is carried in thousandths so the scheduler can compare it without
// floating point, matching the existing GLM-5.2 drafter's convention.
#define K3_DSPARK_CONFIDENCE_MILLI_ONE 1000u

static_assert(K3_DSPARK_AUX_LAYER_COUNT <= K3_LAYERS,
	"a tap cannot name a layer the target does not have");
static_assert(K3_DSPARK_MASK_TOKEN_ID < K3_VOCAB,
	"the mask token is a real vocabulary id");
