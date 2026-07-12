# GLM-5.2 Measured-Status Deployment Observation

```text
commit:      e459d41df92f3bdaf2f8265e66151f97249c46f0
release:     glm52-fp8-main-e459d41-b1-measured-status
generation:  20260712063211
max active:  1
KV pool:     1048576 tokens
```

All requests used `/v1/completions`, `temperature=0`, and streaming SSE. The
API credential is omitted. No performance claim is made from these files.

The clean-start known prompt returned the wrong token. The immediate identical
repeat returned the expected token. This is a measured determinism failure, not
an accuracy pass.
