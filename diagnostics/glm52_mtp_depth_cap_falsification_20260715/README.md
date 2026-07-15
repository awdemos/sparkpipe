# GLM52 B1 MTP depth-cap falsification

This directory retains the raw end-to-end ring receipts that falsified the
adaptive MTP draft-depth cap of two. Both runs used the same eight technical
prompts, greedy decoding, one active request, and 64 emitted tokens per prompt
on the 13-rank FP8 ring.

## Result

| Release | Draft limit | Tokens | Elapsed | Throughput |
| --- | ---: | ---: | ---: | ---: |
| `ebb32153d7c1a04865208b42d154a2dd67d4f47b` | 6 | 512 | 72.507460 s | 7.061342 tok/s |
| `b50310375fd345dc245a851d912ecde4904d2825` | 2 | 512 | 78.659700 s | 6.509051 tok/s |

The depth-two cap reduced throughput by 7.821337 percent. All eight pairs of
64-token output sequences were token-for-token identical. That establishes
output continuity for this comparison; it is not a formal model-accuracy
measurement, so accuracy remains `NOT_MEASURED`.

The uncapped trace recorded:

| Depth | Proposed | Accepted | Survival |
| ---: | ---: | ---: | ---: |
| 1 | 225 | 187 | 0.831111 |
| 2 | 177 | 82 | 0.463277 |
| 3 | 75 | 7 | 0.093333 |
| 4 | 5 | 0 | 0.000000 |

The capped trace recorded 230 verify dispatches, with 190/230 depth-one and
82/180 depth-two survivals. Removing depth three increased target-model work
enough to outweigh the saved draft work.

## Corpus mapping

`prompts.tsv` is the corpus of record. The uncapped files use indices 0-7.
The depth-two run included two additional exploratory prompts, so the matched
indices are `0,1,3,4,5,7,8,9`.

## Command shape

Each prompt was run with:

```text
python3 tools/sparkpipe_api_stress.py --url http://127.0.0.1:18080/v1/completions --health-url http://127.0.0.1:18080/health --model glm-5.2 --requests 1 --concurrency 1 --max-completion-tokens 64 --temperature 0 --stream --timeout-s 180 --health-interval-s 0.1 --prompt <prompt> --output-jsonl <prompt.jsonl> --summary-json <prompt.summary.json> --full-output
```

`baseline_depth6/` and `depth2/` contain the unmodified JSONL, summaries, and
MTP trace logs collected on Spark0. The trace is enabled by
`SPARKPIPE_PP13_TRACE=1` and exposes proposed and accepted depth per verify
dispatch.
