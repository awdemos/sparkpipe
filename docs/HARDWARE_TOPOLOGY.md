# Hardware topology

One file per deployment describes the hardware; everything else is a
projection of it. The schema is `schema/hardware_topology.schema.json`, the
facts live in `examples/topologies/*.json`, and
`tools/generate_topology.py` validates the facts and derives the consumers'
formats. Edit a topology JSON and rerun the generator; never edit a generated
file.

## What a topology file says

- **node_types[]** - named hardware characteristics. `spark` is the default
  entry (GB10: 273 GB/s unified memory bandwidth, BF16/FP8/FP4 TFLOP/s per
  docs/GB10_CUDA_COST_MODEL_CALIBRATION.md, 128 GB unified memory, internal
  NVMe GB/s, 2 network ports), but any node type is definable the same way -
  a deployment that is not all sparks adds its own entry and references it
  from `compute_nodes[].node_type`. Every number carries a `provenance`
  string saying whether it is measured, spec-derived, or assumed.
- **compute_nodes[]** - the static per-node facts: node type, rank, one IPv4
  address per cabled fabric port, the NVMe device path, and optional
  per-node `overrides` (consumed at the JSON level by deployment tooling;
  not projected into the C tables).
- **fabrics[] + ports** - the network. Three modes, matching what the
  hardware supports and what `ring/transport/fabric_topology.c` validates:
  - `ring` - direct point-to-point links, 1-2 ports per node (two in
    practice; one is a degraded bring-up). Example:
    `examples/topologies/ring_13node_bringup.json`, the 13-node July GLM
    bring-up ring (spark0..sparkc, port 0 counter-clockwise, port 1
    clockwise, one /30 per link).
  - `single_switch` - one 100 Gbps all-to-all switch, exactly one port per
    node. Example: `single_switch_16node.json`.
  - `dual_switch` - two 100 Gbps switches, exactly two ports per node, port
    0 on switch A and port 1 on switch B. The production end state. Example:
    `dual_switch_16node_production.json`. The runtime still fails closed on
    this mode until dual-rail ownership and failover are qualified.

## What the generator emits

```
python3 tools/generate_topology.py           # write
python3 tools/generate_topology.py --check   # fail if anything is stale
```

- `deployment/include/sparkpipe/spark_hardware_topology.h` and
  `deployment/src/spark_hardware_topology_tables.c` - the topology as
  compile-time tables: node types, fabrics, per-node ports with their fabric
  indices and addresses, ranks, and the flat per-mode projection
  (rail/switch/port counts, link speed, MTU, debug/future flags) that a
  consumer copies into a `SparkFabricTopologyConfiguration`. Every shipped
  topology is one exported symbol, all of them linked from
  `spark_hardware_topology_registry`.
- `examples/runtime_configs/{ring_debug_single_rail,single_switch_100g,
  dual_switch_dual_rail_future}.json` - the phase-6 runtime configs. They
  predate the topology files and are now pure projections, regenerated
  byte-identically; the topology file's `runtime_projection` block owns
  their mode, extras, and description.

## What validation enforces

Beyond the schema (required fields, value ranges, the mode enums), the
generator refuses a topology when: a node references an unknown node type or
fabric; a fabric endpoint lacks a valid, globally unique IPv4 address; ranks
are not exactly 0..N-1; a node's port count violates its mode (ring 1-2,
single switch 1, dual switch 2) or exceeds its node type's port count; a
dual-switch node's two ports land on the same fabric; fabric kinds disagree
with the mode (ring = direct, switched modes = switch); a declared fabric
has no endpoints; or the used fabrics disagree on speed/MTU (no flat runtime
projection exists). `tests/test_hardware_topology.py` proves each of these
rejections fires, that the generated files are fresh, and that the generated
C compiles under `-Wall -Wextra -Werror`; it is wired into `tools/gates.sh`
as the "hardware topology" gate.
