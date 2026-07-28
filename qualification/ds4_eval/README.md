# ds4-eval retained results

This directory is SparkPipe's canonical record for complete `ds4-eval` runs.
The embedded 92-question set is a capability regression suite, not an official
GPQA, SuperGPQA, AIME, or security leaderboard. The upstream description and
protocol are pinned at
[`antirez/ds4@54b36ed`](https://github.com/antirez/ds4/blob/54b36ed9ba42da31b24f2d1a5feb075c2475dbb1/README.md#capability-evaluation).

## Highest retained result

| Model | Execution | GPQA | SuperGPQA | AIME2025 | COMPSEC | Overall | Exact archive |
|---|---|---:|---:|---:|---:|---:|---|
| Kimi K3 | Kimi coding API, temperature 1.0, 16,000-token limit | 23/25 | 21/25 | 21/25 | 16/17 | **81/92** | [`runs/kimi-k3-api-20260728`](runs/kimi-k3-api-20260728/) |

This is the highest result for which this repository retains the complete run.
It is not claimed as an all-time record because the earlier project results
were not archived.

For context, a separately published DeepSeek V4 Flash 4Expert Q4_K run reports
80/92 (AIME 20/25, GPQA 22/25, SuperGPQA 22/25, COMPSEC 16/17). SparkPipe does
not retain that run's responses, and it is not treated as a locally audited
entry. See the
[pinned model-card report](https://huggingface.co/cloudyu/DeepSeek-V4-Flash-4Expert-GGUF/blob/39a195539a9ad27a28dc70b86f32a1d8651af929/README.md#gguf-evaluation-report--4expert-q4_k-gguf-by-ds4-eval).

## Historical runs awaiting artifact recovery

The project previously evaluated these models, but no numeric score file or
exact response archive was found in SparkPipe history, local worktrees,
downloaded snapshots, or upstream `ds4` history:

- Qwen 3.6 27B — remembered as the prior project high score.
- Gemma — remembered as a competitive result.
- DeepSeek V4 — remembered as part of the comparison set.

Do not fill in their scores from memory. Add them to the retained table only
when the original summary or response trace is recovered.

## Kimi K3 archive

[`runs/kimi-k3-api-20260728`](runs/kimi-k3-api-20260728/) contains:

- `REPORT.md`: human-readable result, protocol, and provenance.
- `INTEGRITY.json`: hashes for the report, manifests, and complete response
  checksum stream.
- `summary.json`: all 92 grades and aggregate usage.
- `cases.json`: exact ordered cases, rendered prompts, source commit, and
  `ds4_eval.c` hash.
- `responses/*.json`: all 92 assembled API responses. Each file preserves the
  content, reasoning, response ID/model, finish reason, token usage, exact
  prompts and prompt hash, request settings, grade, source pin, and timings.

The archive preserves exact assembled response text. It does not preserve raw
SSE event bytes, chunk boundaries, or HTTP headers. The response-file checksum
stream, using paths relative to the run directory, is:

```text
f074d5003dcc3b83f1554e28210461b9ea69c4189c3ad6038d3bb0dc2d36c7e1
```

The stream is SHA-256 over 92 filename-sorted lines of:

```text
<file SHA-256><two spaces>responses/<filename><LF>
```

Validate the retained run:

```sh
python3 qualification/ds4_eval/compare_runs.py \
  qualification/ds4_eval/runs/kimi-k3-api-20260728
```

Compare a future normalized run:

```sh
python3 qualification/ds4_eval/compare_runs.py \
  qualification/ds4_eval/runs/kimi-k3-api-20260728 \
  /path/to/candidate-run
```

Validation independently re-extracts and regrades every retained output and
fails closed on checksum drift, missing cases, changed prompts, source drift,
malformed output, or mismatched case identity. Comparison reports score and
family deltas, pass/fail transitions, extracted-answer changes, and exact
content/reasoning match counts with hashes for changed cases.

## Four-bit quality protocol

The Kimi run is a sampled capability reference. It used temperature 1.0 with no
seed, so exact textual equality against it is not a valid quantization gate.

For the locally hosted four-bit model:

1. Run the same pinned 92 cases and archive every exact response.
2. Compare overall and per-family scores against this retained 81/92 reference.
3. Compare local full precision against local four-bit with identical settings,
   `--temp 0 --seed 1`, and native byte-counted `--trace` files. Any source,
   prompt, token-limit, seed, or sampling mismatch is a configuration error.
4. Use upstream's official-continuation target-token NLL test as the direct
   quantization-quality gate. A sampled answer score alone cannot establish
   that four-bit weights preserved the full model distribution.
