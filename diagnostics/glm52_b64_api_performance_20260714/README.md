# GLM52 B64 API performance

Date: 2026-07-14

This directory retains the first measured API throughput sweep after the FP8
active-row fix and byte-exact 13-rank equivalence gate. The deployed runtime
contained no stage dumps, phase hashes, completion debug logs, or PP13 packet
tracing.

## Identity

```text
runtime commit: 1d7f176b7c0cb4277d55fa914db4b6472de2de56
release: glm52-fp8-main-1d7f176b-b64-perf
generation: 20260713220928
benchmark client commit: 989c345d
quantization: FP8
PP13 ranks: 13
configured maximum active sequences: 64
KV pool tokens per rank: 16384
KV logical blocks: 256
MTP: disabled
DSpark: disabled
```

`installed_hashes.txt` records the resident, builder, driver, and transport
hashes from every rank. They match `release_manifest.json`. `gpu_inventory.txt`
records the GPU, driver, and memory reported by every rank.

## Correctness gates

The exact six-layer SM121 archive and linked driver both processed the retained
25-token input with B64 graph replay and matched the serialized stage-0 oracle
byte for byte:

```text
expected_output_match=1
submissions=25
total_us=385741.569
maximum_us=30619.746
graph_captures=1
graph_replays=25
```

After deployment, `Say OK. OK.` returned token 10397, text `" OK"`, followed by
a done event. PR #359 contains the 300/300 serialized stage-boundary identity
receipt for the same active-row runtime change.

## API results

All requests used the OpenAI-compatible completions endpoint, greedy decoding,
streaming responses, a simultaneous launch barrier, and a 300-second timeout.
Every request returned one done event. B1 requested 32 output tokens; the other
cohorts requested 16 output tokens per request.

```text
cohort  live lanes  tokens  elapsed_s  token/s  avg TTFT_s  avg request_s
B1      1           32      7.9073     4.0469   0.4591      7.9041
B4      4           64      8.4442     7.5792   0.8666      8.4416
B16     16          256     12.1957    20.9909  1.2013      12.1871
B64     64          1024    17.5887    58.2192  1.6778      17.5674
```

The B1 post-first-token rate derived from the retained request timing is 4.16
token/s. The live health counters observed maximum prefill and decode lane
counts of 64 during B64, so these are actual batched cohorts rather than B1
requests serialized by the gateway.

The B1 prompt was a single technical continuation. B4, B16, and B64 cycled the
six built-in technical prompts from `sparkpipe_api_stress.py`; prompt token
counts therefore vary slightly across those requests.

## Commands

The four runs used this shape, with `N` set to 1, 4, 16, or 64 and output-token
count set to 32 for B1 and 16 otherwise:

```sh
AUTH_TOKEN_FROM_INSTALLED_FILE=$(cat /home/spark0/sparkpipe_runtime/API_KEY) \
python3 tools/sparkpipe_api_stress.py \
    --url http://127.0.0.1:18080/v1/completions \
    --health-url http://127.0.0.1:18080/health \
    --requests N --concurrency N \
    --max-completion-tokens TOKENS \
    --api-key-env AUTH_TOKEN_FROM_INSTALLED_FILE \
    --temperature 0 --stream --timeout-s 300 \
    --health-interval-s 0.25
```

The stress client at PR #361 reads SSE responses line by line. The prior 4 KiB
buffered reader overstated TTFT; no result from that reader is included here.

## Scope

These are measured end-to-end throughput and latency results for short-context
FP8 inference without MTP or DSpark. They are not long-context, JIT-KV, MTP,
DSpark, B256, or B1024 claims. Model quality beyond the retained serialized
equivalence prompt remains `NOT_MEASURED`.

After the B64 run, health reported zero live requests, zero queued requests,
zero event backlog, zero dropped events, and no blocker.
