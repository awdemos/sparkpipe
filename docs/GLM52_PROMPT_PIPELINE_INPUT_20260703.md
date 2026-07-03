# GLM52 Prompt Pipeline Input

`tools/glm52_prompt_pipeline_input.py` is the current prompt-facing bridge into
the Spark2 local GLM52 pipeline.

It supports:

```text
--prompt "..."
--prompt-file path
--token-ids 1,2,3,4
--chat
--run-pipeline
```

The script writes:

```text
prompt.txt
prompt_tokens.txt
prompt_tokens.json
prefill_plan.json
prefill_chunks.jsonl
pipeline_env.sh
```

The script now emits two different things:

```text
1. full-prompt prefill plan artifacts
2. narrow compatibility environment for the current local validation pipeline
```

The prefill plan describes the production-shaped prompt work:

```text
prompt token ids [0..N)
    -> prefill chunks over token ids [0..N)
    -> first decode step begins after the full prompt is resident
```

By default the chunk size is:

```text
256 tokens per prefill chunk
16 tokens per KV/prefix-cache block
```

For a 41-token prompt and `--prefill-chunk-tokens 16`, the emitted chunk
manifest is:

```text
chunk 0: offset 0,  token_count 16
chunk 1: offset 16, token_count 16
chunk 2: offset 32, token_count 9, final_prefill_chunk=1
first decode step: after full prompt prefill
```

That is the shape the C request API and scheduler need for:

```text
arbitrary-length prompt tokens
    -> chunked prefill
    -> KV block table reservation
    -> decode continuation
```

The C-side dry-run checker consumes the token file and proves the request API
turns it into prefill dispatches:

```sh
build/sparkpipe_glm52_prefill_dryrun \
    --tokens build/glm52_prompt_prefill_chunk_smoke/prompt_tokens.txt \
    --max-prefill-tokens 16
```

For a 21-token prompt, the expected sequence is:

```text
0   prefill       offset 0    token_count 16   remaining 5
1   prefill       offset 16   token_count 5    remaining 0
2   decode_ready
```

When `GLM52_PREFILL_TOKEN_IDS_FILE` is present, the Spark2 local pipeline gate
runs that same C checker before the decode/tail-window compatibility pipeline:

```text
tools/glm52_spark2_local_pipeline_gate.sh
    -> build/sparkpipe_glm52_prefill_dryrun
    -> prefill_schedule.tsv
    -> require final row kind=decode_ready
    -> run current CUDA local pipeline
```

The gate exports:

```text
glm52_local_pipeline_prefill_schedule=<path>
glm52_local_pipeline_prefill_steps=<count>
glm52_local_pipeline_prefill_tokens=<count>
```

For prefill schedule testing without launching CUDA stages:

```sh
GLM52_LOCAL_PIPELINE_PREFILL_ONLY=1 \
GLM52_PREFILL_TOKEN_IDS_FILE=build/glm52_prompt_prefill_chunk_smoke/prompt_tokens.txt \
tools/glm52_spark2_local_pipeline_gate.sh
```

This is still not claiming CUDA prefill execution. It is a fail-closed bridge
that makes the local pipeline consume and check the SparkPipe C prefill schedule
instead of ignoring prompt prefill artifacts.

The current local execution bridge is still intentionally narrow:

```text
full prompt text
    -> local HF tokenizer or explicit token ids
    -> persisted full token-id artifact
    -> persisted chunked prefill plan artifact
    -> last four tokens become the current validation context
    -> first three tail tokens feed prefill/KV
    -> final tail token becomes GLM52_LOCAL_PIPELINE_INPUT_TOKEN_ID
    -> existing local pipeline gate runs dense-prefix/routed decode
```

The script refuses to run prompts longer than the four-token validation window
through `--run-pipeline` unless `--allow-tail-window-run` is passed. This keeps
the local compatibility path from reporting a fake full-prompt pass.

This is not yet full arbitrary-length prompt prefill execution. The current
validation context is four tokens, so the compatibility path uses a tail window.
This is separate from the production-shaped prefill plan, which covers the full
prompt:

```text
prompt token ids [0..N)
    -> production prefill chunks [0..N)
    -> compatibility validation context [N-4..N)
```

That is still a real improvement over the previous fake prefill behavior, where
the validator invented prior tokens with:

```text
input_token_id - 3
input_token_id - 2
input_token_id - 1
```

The artifact preserves the full token list so the next production step can wire:

```text
prompt token ids
    -> prefill_plan.json / prefill_chunks.jsonl
    -> arbitrary-length chunked prefill frames
    -> KV block table
    -> stage-slice prefill
    -> decode continuation
```

Python is allowed here only as setup/bringup glue. The production target is a C
tokenizer artifact plus C/CUDA request ingestion:

```text
tokenizer.model/tokenizer.json
    -> generated C tokenization tables
    -> SparkPipe request API prompt-token buffer
    -> prefill/decode scheduler
```

The prompt bridge refuses to report a fake full-prompt pass. It labels the
current mode as:

```text
tail_window_prompt_prefill_validation_context
```
