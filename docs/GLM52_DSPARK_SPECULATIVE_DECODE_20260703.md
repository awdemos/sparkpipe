# GLM-5.2 DSpark speculative decode integration

This pass adds an internal Sparkpipe DSpark speculative-decode path for the
GLM-5.2 FP8 verifier deployment shape.

The supported DSpark contract is intentionally exact:

```text
auxiliary verifier layers: 8, 23, 39, 55, 70
verifier quantization: FP8 E4M3
draft dtype: BF16
fixed PP13 stage width: 6 layers
speculator draft layers: 5
block size: 8
maximum speculative verify tokens: 7
full vocabulary size: 154880
draft intermediate size: 12288
draft attention heads: 64
draft KV heads: 64
draft head dimension: 64
markov rank: 256
max anchors: 1024
confidence head: required by policy
markov head: required by policy
```

The request API now rejects DSpark configuration unless the attached
`SparkGlm52DsparkSpeculator` validates this exact model contract. In particular,
the optional path is not allowed to silently run against a non-FP8 verifier.

## Setup manifest

The downloaded Hugging Face checkpoint can be reduced to a small setup-time
manifest with:

```sh
python3 tools/glm52_dspark_manifest.py \
  --model-dir /home/spark0/ds4_nvme/models/hf/RedHatAI/GLM-5.2-speculator.dspark \
  --output /home/spark0/ds4_nvme/sparkpipe_artifacts/dspark/glm52_dspark_manifest.json
```

The tool validates the checkpoint `config.json` against the Sparkpipe DSpark
contract and records the `model.safetensors` byte count and SHA-256. Python is
only used here during one-time setup; serving should consume the JSON contract
and the resident weights from C/CUDA.

The PP13 tap mapping is:

```text
layer 8  -> stage 1, layers 6..11
layer 23 -> stage 3, layers 18..23
layer 39 -> stage 6, layers 36..41
layer 55 -> stage 9, layers 54..59
layer 70 -> stage 11, layers 66..71
```

## Runtime path

The request API now owns the speculative loop:

```text
normal decode dispatch
    -> DSpark tap-capture flag
    -> verifier hidden taps become ready
    -> DSpark draft backend produces proposed tokens and confidence
    -> request moves to READY_SPECULATIVE_VERIFY
    -> scheduler packs equal-length draft verify lanes
    -> speculative verify dispatch carries draft token ids/confidences
    -> completion commits accepted draft tokens and fallback token
    -> request either completes or immediately prepares the next draft
```

Callers do not manage speculative decoding. They submit the same request shape
as before, plus optional request flags such as realtime or disable speculation.
Sparkpipe decides whether DSpark is enabled from internal policy.

## New source files

```text
include/sparkpipe/spark_glm52_dspark.h
src/spark_glm52_dspark.c
tests/test_glm52_dspark.c
```

## Request API changes

New configuration flag:

```c
SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE
```

New per-request opt-out flag:

```c
SPARK_GLM52_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION
```

New states:

```c
SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY
SPARK_GLM52_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY
```

New dispatch kind:

```c
SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH
```

New dispatch flags:

```c
SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE
SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY
SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_CONFIDENCE_TRUNCATED
```

The speculative verify dispatch carries draft token ids and confidence values
per lane. Completion writes accepted/committed counts back into the same dispatch
object before calling `SparkGlm52RequestApiCompleteDispatch`.

## Driver / resident module changes

The model-driver frame flags now include:

```c
SPARK_MODEL_DRIVER_FRAME_FLAG_DSPARK_TAP_CAPTURE
SPARK_MODEL_DRIVER_FRAME_FLAG_SPECULATIVE_VERIFY
```

The generated driver admission path no longer rejects speculative-verify frames
just because `new_token_count` exceeds the normal decode cap.

The resident decode-stage frame context now has a first-class DSpark hidden-tap
contract:

```c
SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_HIDDEN_TAPS
```

When the flag is set, firmware validates the exact GLM-5.2 DSpark tap plan and
requires BF16 output buffers for all five auxiliary taps.

## Current boundary

This pass intentionally does not pretend to load or execute the 4B DSpark
safetensors draft network. The runtime, scheduler, request API, frame flags,
manifest contract, and resident tap ABI are now in place. The remaining
production work is a C/CUDA DSpark draft backend under
`SparkGlm52DsparkDraftFunction`, fed by the five resident device taps and the
setup manifest.
