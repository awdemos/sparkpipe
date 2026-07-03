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
pipeline_env.sh
```

The current execution bridge is intentionally narrow:

```text
full prompt text
    -> local HF tokenizer or explicit token ids
    -> persisted full token-id artifact
    -> last four tokens become the current validation context
    -> first three tail tokens feed prefill/KV
    -> final tail token becomes GLM52_LOCAL_PIPELINE_INPUT_TOKEN_ID
    -> existing local pipeline gate runs dense-prefix/routed decode
```

This is not yet full arbitrary-length prompt prefill. The current validation
context is four tokens, so long prompts use a tail window:

```text
prompt token ids [0..N)
    -> context token ids [N-4..N)
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
    -> arbitrary-length prefill frames
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
