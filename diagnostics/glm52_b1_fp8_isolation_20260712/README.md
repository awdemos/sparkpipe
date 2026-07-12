# GLM-5.2 FP8 B1 isolation receipt

## Build identity

- Git commit: `6d36eb40fb248498ee6bff7a760524c15d7b33a0`
- Release: `glm52-fp8-main-6d36eb4-b4-queue-1024`
- Release generation: `20260712104856`
- Quantization: FP8 E4M3
- Prompt token ids: `45494 10397 13 10397 13`

The CUDA validator in this change was built from that commit plus the exact
full-vocabulary validator fix. The deployed resident runtime remained the
immutable release above.

## Measured results

### Isolated stage 0

The current FP8 six-layer stage-0 serial sequence was compared numerically
with the committed official stage-0 oracle. Across the five prompt positions,
the final layer had maximum relative L2 error `0.03027` and minimum cosine
similarity `0.999542`.

The current FP8 B1 decode path was also compared with the prior slow,
known-accurate FP8 implementation:

- maximum absolute error: `0.001953125`
- relative L2 error: `0.0249448`
- cosine similarity: `0.999689`

Stage-0 graph replay was byte-identical to its eager result for the same B1
input.

### Exact serialized PP13 chain

All thirteen exact six-layer stages ran serially with the output of each stage
fed into the next stage. Each stage used the rank-local FP8 stage payload that
the ring release uses. Five prompt positions completed at every stage.

The final stage selected full-vocabulary token `10397`, which is the expected
token for this fixture (`" OK"`). The final stage processed the five serial
positions in `114607.652 us`; its maximum single submission was
`31710.691 us`.

The validator defect fixed here was concrete: the exact final-stage fixture
advertised a full-vocabulary epilogue while loading only 256 LM-head rows and
256 token ids. The production kernel indexed the declared 154880-token
vocabulary and faulted. The fixture now loads all 154880 LM-head rows and an
identity token-id table, then checks the expected full-vocabulary token when
`GLM52_EXACT_PP13_EXPECTED_FULL_VOCAB_TOKEN` is set.

### Live ring B1 smoke

After a clean all-resident restart followed by rank daemons and then the
gateway, the same immutable release produced:

- an earlier clean-start cold request selected token `10397` and completed in
  about `0.85 s`
- its immediate warm repeat selected token `10397` and completed in about
  `0.55 s`
- eight-token request: `10397 13 10397 13 10397 13 10397 13`
- decoded text: `" OK. OK. OK. OK."`
- eight-token request wall time: about `2.746 s`
- post-run health: no live or queued requests and no event backlog

After a later full restart, the first request selected token `13` while the
next three identical requests selected token `10397`. The hidden hash chain
localized the first divergence to rank 1, position 1. Rank 1 executed the cold
request in position order `1,2,3,4,5,0`; the warm requests executed
`0,1,2,3,4,5`. Position 0 had been deferred while its downstream work socket
connected, and the queue allowed later positions from the same sequence to
overtake it. This change prevents a queued predecessor from being overtaken
without serializing unrelated requests.

The earlier successful requests prove the known B1 fixture can traverse the
ring. The cold-ordering failure means reliable ring B1 remains pending until
the queue fix is deployed and measured. This is not a model-wide accuracy
measurement.

## Remaining measured gaps

- Model-wide FP8 prompt accuracy: `NOT_MEASURED`
- Long-context B1 accuracy: `NOT_MEASURED`
- MTP correctness and acceptance: `NOT_WORKING`
- DSpark correctness and acceptance: `NOT_WORKING`
- B1 final-stage graph replay: two eager and two graph runs produced identical
  hidden SHA-256 `a609cf09f6852e4cb7032a60ded8aba4fb61993a8cf52469301483bd96b49e98`
  and selected token `279` for the isolated token-0 input
- Restart robustness: a downstream resident restart can drop the upstream
  sender's first frame before reconnect; clean startup order currently avoids
  this

The live ring's hidden hashes are not byte-identical to the eager serialized
chain even though the known final token matches. That difference requires a
numeric dump comparison before it can be classified as benign rounding or a
remaining execution-path discrepancy.
