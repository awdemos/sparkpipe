# GLM-5.2 FP8 B1 exact-path receipt

## Scope

This receipt covers the base FP8 B1 path only. It does not establish corpus
accuracy, MTP, DSpark, long-context DSA, JIT KV, multi-sequence batching, or
production performance.

The deployed release was:

```text
commit: 029aaf76b77c57d8291acfed645ff36e7d79aeeb
release: glm52-fp8-main-029aaf7-b1-exact-oracle
generation: 20260712141811
max active sequences: 1
KV pool tokens: 1048576
```

PR #314 fixed the exact FP8 validator. It previously loaded and bound FP8
dense weights, then silently changed dense MLP execution from quantized mode 3
to legacy mode 1. The production ring stayed on mode 3, so comparisons against
that oracle measured two different execution paths. The validator now keeps
quantized dense execution selected.

## Exact comparison

The matched token sequence was:

```text
45494 10397 13 10397 13
```

The sixth row repeats token `13`, matching the serving engine's first decode
step at sequence position 5. Each rank ran its own six layers in isolation with
its rank-local FP8 `.spstage` and `.spfp8` files. The 60 KiB hidden sequence was
then passed to the next rank's isolated validator.

The deployed ring ran the same six positions with raw hidden dumps enabled.
`boundary_compare.tsv` compares all transmitted boundaries:

```text
12 ranks x 6 positions = 72 comparisons
byte mismatches = 0
```

The isolated final rank selected token `10397`. The deployed ring also emitted
token `10397`, text `" OK"`, followed by a done event.

## Reliability observations

Three consecutive greedy requests for `Say OK. OK.` each emitted token `10397`
and completed. An eight-token request emitted:

```text
10397 13 10397 13 10397 13 10397 13
```

The constrained factual prompt `Question: What is the capital of France?` with
an `Answer:` suffix began with token `12089`, text `" Paris"`.

Post-run health reported zero live requests, zero queued requests, zero event
backlog, and zero dropped events. It correctly retained
`accuracy_status=NOT_MEASURED` and `performance_status=NOT_MEASURED`.

## Artifact layout

- `serialized/`: thirteen isolated six-layer outputs, each containing six BF16
  hidden rows.
- `ring/`: the 72 transmitted hidden rows from ranks 0 through 11.
- `boundary_compare.tsv`: per-rank, per-position SHA-256 equality.
- `artifact_hashes.tsv`: installed driver and builder SHA-256 on all 13 Sparks.
- `api/`: raw streaming API and post-run health receipts.
- `sha256.txt`: checksums for the retained evidence files.

The SM121 package gate also passed for both the raw archive and linked driver.
Each executed six layer bodies with one graph capture and two replays. The
linked run produced 6,143 nonzero BF16 values with checksum
`5958189933213524842`; its timed stage submission was 14,455.675 microseconds.

## Honest status

Base FP8 B1 stage equivalence for this fixture is `OBSERVED`. End-to-end smoke
behavior for the retained prompts is `OBSERVED`. Broad model accuracy is
`NOT_MEASURED`. MTP, DSpark, JIT KV, and multi-sequence batching remain
`NOT_WORKING` in the deployed health contract.
