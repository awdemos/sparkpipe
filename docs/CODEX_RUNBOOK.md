# Codex Runbook

This is the workflow Codex should follow for SparkPipe work. Do not invent a
new path unless a command here fails for a new, concrete reason.

## Goal

Get GLM-5.2 inference working on real prompts at production speed. A merge, a
report, a local test, or a validation fixture is only useful when it moves that
goal forward.

## Non-Negotiables

- Main is the advisor handoff branch.
- Use PRs. Do not push directly to `main`.
- Validate deployment behavior only from pulled `main` on the target Spark.
- Do not validate CUDA behavior on the Mac.
- Do not use local hotpatches, dirty trees, copied source files, or unmerged
  branches as production evidence.
- Do not delete or regenerate expensive `.sp*` packs unless the user confirms.
- Do not use compatibility fallback chains in production paths.
- Prefer `make -j`; use `-j1` only for a specific race or log-order diagnosis.

## Repo Locations

```text
local writable worktree:
    /Users/mac/Documents/New project 4/sparkpipe-route-slice

local main checkout:
    /Users/mac/sparkpipe

Spark2 main checkout:
    /home/spark2/src/sparkpipe-main-live

Spark2 stable GLM52 B12x pack root:
    /home/spark2/sparkpipe_artifacts/glm52_b12x_resident_moe_all_v3

Spark2 local model root:
    /home/spark2/models/hf/nvidia/GLM-5.2-NVFP4
```

## Git Workflow

1. Start from current main in the writable worktree.

```sh
git -C "/Users/mac/Documents/New project 4/sparkpipe-route-slice" fetch origin main
git -C "/Users/mac/Documents/New project 4/sparkpipe-route-slice" checkout -B codex/<short-task> origin/main
```

2. Make the smallest production-relevant change.

3. Run local non-CUDA tests.

```sh
cd "/Users/mac/Documents/New project 4/sparkpipe-route-slice"
make -j test
```

4. Commit.

```sh
git -C "/Users/mac/Documents/New project 4/sparkpipe-route-slice" add <files>
git -C "/Users/mac/Documents/New project 4/sparkpipe-route-slice" commit -m "<message>"
```

5. Push with the SparkPipe PAT as HTTPS basic auth. The `.env` file contains
   `GITHUB_PAT`. Do not print the token. Do not use bearer auth for git push.

```sh
cd "/Users/mac/Documents/New project 4"
. /Users/mac/sparkpipe/.env
auth_header="$(printf 'x-access-token:%s' "$GITHUB_PAT" | base64 | tr -d '\n')"
git -C sparkpipe-route-slice \
    -c "http.https://github.com/.extraheader=AUTHORIZATION: Basic $auth_header" \
    push -u origin codex/<short-task>
```

If HTTPS push fails, check permission before declaring a blocker:

```sh
GH_TOKEN="$GITHUB_PAT" gh repo view sparkpipe/sparkpipe --json viewerPermission,defaultBranchRef
```

6. Create the PR with the same PAT.

```sh
GH_TOKEN="$GITHUB_PAT" gh pr create \
    --repo sparkpipe/sparkpipe \
    --base main \
    --head codex/<short-task> \
    --title "<title>" \
    --body "<summary and tests>"
```

7. Merge the PR to main.

```sh
GH_TOKEN="$GITHUB_PAT" gh pr merge <number> \
    --repo sparkpipe/sparkpipe \
    --merge \
    --delete-branch
```

8. Pull merged main locally and on Spark2.

```sh
git -C /Users/mac/sparkpipe pull --ff-only origin main
ssh spark2 'cd /home/spark2/src/sparkpipe-main-live && git pull --ff-only origin main'
```

## Spark2 Build And Test Workflow

Use Spark2 for CUDA and runtime evidence.

```sh
ssh spark2 'cd /home/spark2/src/sparkpipe-main-live && make -j test'
```

Run the exact local pipeline gate from pulled main:

```sh
ssh spark2 'cd /home/spark2/src/sparkpipe-main-live && PATH=/usr/local/cuda/bin:$PATH make -j glm52_spark2_local_pipeline_gate'
```

The gate must use local Spark2 storage. It must not read the model from Mac
mounts, spinning disks, or remote cold storage.

Expected summary file:

```text
/home/spark2/src/sparkpipe-main-live/build/glm52_local_pipeline_gate/local_pipeline_summary.tsv
```

## Performance Evidence

Report timings by path:

- decode stage-slice timing
- final-token and MTP timing
- prefill timing
- transport timing

Do not collapse those into one number. For stage timings, report the slowest
stage and the filled-pipeline ceiling:

```text
filled_pipeline_tok_per_s = batch_size * 1000 / slowest_stage_ms
```

## Prompt Inference Rule

Do not call last-token embedding decode "prompt inference." Real prompt
inference requires:

- tokenizer or token-id input
- prompt prefill through all 78 layers
- KV ownership retained across prefill and decode
- exact PP13 decode loop for generated tokens
- token-id to text output

If any of those is missing, fail closed and name the missing component.

## Pack And Artifact Rules

Stable `.sp*` packs are source-like compiled firmware artifacts. They are
expensive and should be preserved.

- Keep GLM-5.2 4-bit and 8-bit packs available for experiments.
- Do not delete source model files or generated packs without user confirmation.
- Use the stable pack root directly; do not add fallback directory chains.
- If a pack path changes, update the code and tests to use the new single path.

## What To Avoid

- No rsync/scp/file handoff in production transport.
- No Mac CUDA testing.
- No fake PASS from validation-only or compatibility modes.
- No broad fallback chain for old artifact roots.
- No direct `main` push.
- No "blocked" answer before trying the documented PAT basic-auth push.
