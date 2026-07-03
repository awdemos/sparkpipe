# GLM52 Prompt Pipeline Input Bootstrap

`tools/glm52_prompt_pipeline_input.py` is the current prompt-facing bridge into
the Spark2 local GLM52 pipeline.

It supports:

```text
--prompt "..."
--prompt-file path
--token-ids 1,2,3
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
    -> final prompt token becomes GLM52_LOCAL_PIPELINE_INPUT_TOKEN_ID
    -> existing local pipeline gate runs one-token bootstrap decode
```

This is not yet full arbitrary-prompt inference because the existing local gate
does not consume all prompt tokens as prefill KV context. The artifact preserves
the full token list so the next production step can wire:

```text
prompt token ids
    -> prefill frames
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
single_token_bootstrap_no_prefill_kv
```
