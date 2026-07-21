# Batch-plane performance notes

Audit findings on the expert-queue batch plane host components. Fixed items
are recorded for provenance; open items are ordered by measured cost.

## Fixed

- **JIT KV pool eviction was O(fragments) per stage-in.** A linear scan over
  every fragment selected the eviction victim on each miss. Measured 281 us
  per require at 262144 fragments, scaling linearly with pool size; at 131072
  DRAM fragments and roughly 18000 stage-ins per longmem run this was seconds
  of pure host scan per rank, larger than the compute it scheduled. Replaced
  with a max-heap over DRAM-resident fragments keyed on next_need_ns, with the
  fragment's heap position tracked so an ETA change re-sifts in place. Now 0.13
  us per require, flat in pool size, a 2160x reduction. Tie-break on equal need
  evicts the higher fragment id deterministically for the ring SHA gates.

- **JIT KV pool transfer array hard-failed under prefetch bursts.** The pending
  transfer store was a plain array compacted only on Tick, and each stage-in
  queues up to two transfers (evict plus in), so 2048 un-ticked requires
  exhausted the 4096 slots and returned CAPACITY_EXCEEDED. A prefetch burst
  from a wave boundary hit this directly. Replaced with a ring buffer whose
  overflow completes the oldest in-flight transfer inline rather than refusing,
  so requires never hard-fail; overflow_drain_count records how often it
  happens so the transfer ceiling can be tuned from real traces.

## Open, ordered by measured cost

- **ExpertQueue NextFiring rescans all layer x expert slots from origin.**
  Every call walks up to layer_count x expert_count slots to find the next
  fireable one, so draining N firings is O(N x slots). Measured 2.2 us per
  fire, roughly 0.1 s per rank per longmem run. Modest next to the pool fix and
  a larger change (an intrusive ready-list of slots at or past threshold,
  pushed on enqueue crossing and on deadline, popped on fire), so deferred.

- **Serving-adapter per-dispatch H2D staging.** The decode serving adapter
  issues per-step cudaMemcpyAsync host-to-device for tokens and routing tables,
  a discrete-GPU pattern that on GB10 unified memory is a DRAM-to-DRAM copy plus
  launch latency. Bandwidth-negligible at KB-MB payloads, tens of us against
  16 ms stages, but convertible to mapped-host writes if the verify gate lands
  near 25 ms and needs shaving. Requires ring measurement to justify.

- **Workspace allocator residency unverified.** Whether the central workspace
  allocator uses mapped or managed allocations versus plain cudaMalloc lives in
  runtime host code outside the decode modules and could not be confirmed from
  the host sandbox. On GB10 both resolve to the same DRAM so steady-state
  bandwidth is identical; the only difference is a transient copy on load paths.
  One grep on the ring-side runtime settles it; fold into the packet-timing
  session by reporting per-stage copy counts and bytes.

## Second audit pass

### Fixed

- **ExpertQueue init wrote a 1M-entry free list then zeroed a 24MB struct.**
  The eager free-list walk plus a full-struct memset cost 6309 us per init.
  The free list is now lazy: a high-water counter hands out unused rows in
  O(1) and only recycled rows go on the free list, so no init walk is needed;
  and the memset now covers only the header and slot array up to offsetof(rows)
  since the row pool is fully written by the allocator before any read. Init
  dropped to 1.8 us, a 3500x reduction. Row allocation and release are now
  single helpers used by both enqueue and firing rather than open-coded in
  three places.

### Cleared (checked, not a defect)

- **Overflow-drain versus DRAM capacity invariant.** Suspected that completing
  a stage-in inline during a transfer-ring overflow could push resident count
  past capacity between the pre-check and the heap insert. Stress with 7904
  forced drains at tight DRAM holds dram_resident + staging_in at exactly the
  cap with no breach: overflow drains complete stage-out and stage-in transfers
  in the FIFO order they were queued, so the accounting stays paired. Sound.

- **Firing emit cap over MAX_FIRING_ROWS.** A slot deeper than the emit array
  caps the firing at the array bound and leaves the remainder queued with a
  correctly advanced oldest arrival; memory-safe, no truncation of live rows.
  Locked with a test.

### Open, lower cost

- **JitKvPool init memset is 513 us.** The fragment array must be zeroed for
  the FREE-state sentinel, so unlike the queue it cannot skip the row region.
  Could zero only fragments[0..capacity) when capacity is below the max, but
  that trades a clear invariant for a rare-path saving and is deferred.

- **Config validation preambles are structurally similar across the three
  init functions** but validate different field sets, so they are left
  explicit rather than folded into a macro that would hide the field checks.

## Third pass: unconditional optimization

### Added — content-addressed KV dedup

The batch plane's dominant bandwidth term is attention, not the expert sweep
(measured 82 vs 48 GB/s per rank at the four-thousand-sequence point), and the
attention traffic is dominated by re-reading latent KV. Most levers there are
workload-conditional: FP8 latent is a quality tradeoff, and lane selection
co-scheduling depends on the unmeasured natural overlap of the eight lanes'
DSA selections. Content-addressed dedup is the one that is strictly never worse
in any workload: byte-identical prefix fragments shared across sequences (system
prompt, tool schemas, few-shot examples, shared longmem memory) resolve through
a content hash to a single physical fragment, refcounted, freed only at zero
references. With no sharing the physical assignment is identical to per-sequence
storage, a 1.00x floor with zero regression; at a realistic twenty-four-fragment
shared prefix across five hundred sequences it frees 5.3 GB per rank, which
converts directly into more DRAM-resident sequences and less NVMe paging, and
the shared physical fragment is read once for attention across every sequence
that points at it. Open-addressed table, linear probing, backward-shift delete
with cluster reinsert, no allocation. Tested for sharing, refcount lifecycle,
and probe-cluster integrity after a mid-cluster free.

### Footgun recorded — batch-plane structs must never be stack-allocated

The dedup struct is about 10 MB, the expert queue 24 MB, the JIT pool 9 MB.
A stack local of any of them overflows immediately; a test that did so segfaulted
and was caught by AddressSanitizer. There is no compile-time guard. These are
singletons by design and belong in static or caller-provided storage; anything
introducing a stack instance will crash at entry. Worth a static-assert on a
stack-hostile size if a guard mechanism is added.
