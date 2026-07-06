# DSpark tap transport contract and the sideband bridge gap

## Finding: no sideband kind has a runtime data plane

The production topology's sideband descriptors (indexshare selected
tokens included) are consumed by nothing outside the topology module
and its test. The hidden transport packet header carries sideband_kind
and sideband_bytes_per_sequence and the payload rides contiguously
after the hidden bytes, but no code bridges topology descriptors to
packet assembly on export, forwards non-consumed payloads across
intermediate hops, or delivers them on import. Index-share selection
propagation across stages therefore also has no transport path in the
event-driven serving layer today.

## This increment: the declarative tap contract, tested

Topology now appends five DSPARK_HIDDEN_TAP sidebands during Build,
one per auxiliary verifier layer (8, 23, 39, 55, 70 - single-sourced
as SPARK_GLM52_DSPARK_AUX_LAYER_IDS_INITIALIZER, previously duplicated
as literals in two files). Export stages resolve from the stage plan
(1, 3, 6, 9, 11 under PP13 18:6), import stage is the final stage,
payload is active_sequence_capacity x 6144 x 2 bytes of BF16 per tap.
AppendSideband is parameterized by payload_bytes and flags; Validate
branches per kind and enforces the tap payload formula. Transport gains
sideband kind constants (NONE / INDEXSHARE_SELECTED_TOKENS /
DSPARK_HIDDEN_TAP). The topology test asserts the exact tap mapping.

## Remaining work, in dependency order

1. Sideband bridge (serves indexshare AND taps): packet assembly on
   each rank consults the topology descriptors for the hop - append
   exported payloads after the hidden bytes, copy through payloads
   whose import stage is downstream, deliver arrivals. Per-hop sideband
   bytes = sum over descriptors with export <= hop < import. Taps
   accumulate toward the final stage; no multi-kind packets needed if
   the region is ordered by descriptor index.
2. Import delivery into the draft: on the final stage the five tap
   regions land in the DsparkDraftBackend tap arena lanes
   (TapOutputPointers provides per-lane destinations and stride).
3. Serving loop in the pp13 service backend against the existing
   request-API contract: set DSPARK_TAP_CAPTURE on decode dispatches,
   stage lanes and call Draft when taps arrive, submit the
   SPECULATIVE_VERIFY_BATCH dispatch as a multi-token verify frame
   (FRAME_FLAG_SPECULATIVE_VERIFY - driver admission already accepts
   it), commit accepted counts through CompleteDispatch.
4. Validation precondition: the MTP end-to-end gate (acceptance > 0,
   committed == greedy) must pass on spark2 before any dspark
   throughput claim; the verify path shares that machinery.
