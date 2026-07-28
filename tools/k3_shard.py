#!/usr/bin/env python3
"""Slice a K3 resident pack into per-rank TP packs.

The single source of truth for how each K3 tensor splits across a TP group,
in the glm52 shard module's shape: classification is by the manifest's field
name, an unclassified name refuses the whole slice rather than guessing, and
every split is checked against the alignment it owes - output splits land on
whole head blocks, input splits on whole MXFP4 groups when the axis is
packed nibbles.

The standard Megatron split for a weight stored [out, in]: output-dimension
shards need no collective, input-dimension shards produce partials the
layer's closing all-reduce combines (SparkTpCollectiveAllReduceSumF32 is that
collective on this ring), replicated tensors load whole. K3's departures
from glm52, each earned by the architecture:

  the low-rank bottlenecks (decay_down, gate_down for both attention kinds,
  mla_q_down) REPLICATE: their 128-wide output is what every rank's up half
  reads in full, and slicing 128 sixteen ways buys nothing but a collective
  the kv_a latent path replicates, which is what keeps the latent KV cache
  identical per rank - TP cannot shard one KV head
  the concatenated gate|up tensors (shared, dense, and every expert's w1)
  slice EACH HALF and re-concatenate per rank, or the SiTU kernel's
  gate-first contract breaks at every rank boundary
  expert w2 input-splits along an axis stored as nibbles with a scale every
  32: the per-rank K must be whole groups, asserted, and the scale plane's
  k_group columns slice with it
"""
import json
import struct
import sys
from pathlib import Path

MAGIC = 0x4B33504B
GROUP = 32


class ShardFailure(RuntimeError):
    pass


REPLICATED = {
    "attn_norm_weight", "mlp_norm_weight", "attnres_attn_weight",
    "attnres_mlp_weight", "router_weight", "router_bias",
    "kda_decay_down_weight", "kda_gate_down_weight", "kda_decay_bias",
    "kda_head_log_scale", "kda_out_norm_weight",
    "mla_q_down_weight", "mla_q_norm_weight", "mla_kv_a_weight",
    "mla_kv_a_norm_weight", "mla_gate_down_weight", "routed_norm_weight",
}
MODEL_REPLICATED = {"model.norm.weight", "model.attnres_out_weight"}
# output-dimension, sliced on whole head blocks: (head elements, row bytes)
OUTPUT_HEADS = {
    "kda_q_weight": ("kda", 1), "kda_k_weight": ("kda", 1),
    "kda_v_weight": ("kda", 1), "kda_q_conv_weight": ("kda", 1),
    "kda_k_conv_weight": ("kda", 1), "kda_v_conv_weight": ("kda", 1),
    "kda_decay_up_weight": ("kda", 1), "kda_gate_up_weight": ("kda", 1),
    "kda_beta_weight": ("kda_head1", 1),
    "mla_q_up_weight": ("mla_q", 1), "mla_kv_b_value_weight": ("mla_v", 1),
    "mla_gate_up_weight": ("mla_v", 1),
}
# input-dimension on whole head blocks, partials summed by the all-reduce
INPUT_HEADS = {"kda_out_weight": ("kda", 1), "mla_out_weight": ("mla_v", 1)}
# plain output rows / degree
OUTPUT_DIM = {"routed_down_weight"}
# plain input columns / degree, all-reduce after
INPUT_DIM = {"routed_up_weight"}
# [gate; up] concatenated: each half output-splits, re-concatenated per rank
CONCAT_OUTPUT = {"shared_w1_weight", "dense_gate_up_weight"}
INPUT_DIM_PLAIN = {"shared_w2_weight", "dense_down_weight"}


def head_block(kind, geo):
    return {"kda": geo["kda_head"], "kda_head1": 1,
            "mla_q": geo["kv_lora"] + geo["rope"], "mla_v": geo["v_head"]}[kind]


def head_count(kind, geo):
    return geo["kda_heads"] if kind.startswith("kda") else geo["heads"]


def slice_rows(raw, rows, rank, degree):
    if rows % degree != 0:
        raise ShardFailure(f"{rows} rows do not split {degree} ways")
    per = len(raw) // rows
    lo = (rows // degree) * rank
    return raw[lo * per:(lo + rows // degree) * per]


def slice_cols(raw, rows, row_bytes, lo_byte, hi_byte):
    out = bytearray()
    for r in range(rows):
        out += raw[r * row_bytes + lo_byte:r * row_bytes + hi_byte]
    return bytes(out)


class Slicer:
    def __init__(self, pack_path, geo, degree, rank):
        raw = Path(pack_path).read_bytes()
        magic, version, length = struct.unpack_from("<IIQ", raw, 0)
        if magic != MAGIC:
            raise ShardFailure("not a K3 pack")
        self.manifest = json.loads(raw[16:16 + length])
        base = 16 + length
        base += (-base) % 64
        self.raw, self.base = raw, base
        self.geo, self.degree, self.rank = geo, degree, rank
        self.config = self.manifest["config"]

    def bytes_of(self, name):
        entry = self.manifest["tensors"][name]
        return self.raw[self.base + entry["offset"]:
                        self.base + entry["offset"] + entry["bytes"]]

    def emit(self, out_path):
        tensors = {}
        payload = bytearray()
        for name in self.manifest["tensors"]:
            sliced = self.route(name)
            pad = (-len(payload)) % 64
            payload += b"\0" * pad
            tensors[name] = {"offset": len(payload), "bytes": len(sliced)}
            payload += sliced
        echo = dict(self.config)
        echo.update({"tp_degree": self.degree, "tp_rank": self.rank})
        manifest = json.dumps({"config": echo, "tensors": tensors},
                              separators=(",", ":")).encode()
        with open(out_path, "wb") as out:
            out.write(struct.pack("<IIQ", MAGIC, 1, len(manifest)))
            out.write(manifest)
            out.write(b"\0" * ((-out.tell()) % 64))
            out.write(payload)
        return tensors

    def route(self, name):
        geo, degree, rank, cfg = self.geo, self.degree, self.rank, self.config
        raw = self.bytes_of(name)
        field = name.split(".")[-1]
        if name in MODEL_REPLICATED or field in REPLICATED:
            return raw
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            return slice_rows(raw, cfg["vocab"], rank, degree)
        if field in OUTPUT_HEADS:
            kind, _ = OUTPUT_HEADS[field]
            heads = head_count(kind, geo)
            if heads % degree != 0:
                raise ShardFailure(f"{name}: {heads} heads over {degree} ranks")
            rows = heads if field == "kda_beta_weight" \
                else heads * head_block(kind, geo)
            return slice_rows(raw, rows, rank, degree)
        if field in INPUT_HEADS:
            kind, _ = INPUT_HEADS[field]
            heads = head_count(kind, geo)
            if heads % degree != 0:
                raise ShardFailure(f"{name}: {heads} heads over {degree} ranks")
            in_bytes = heads * head_block(kind, geo) * 2
            per = in_bytes // degree
            return slice_cols(raw, len(raw) // in_bytes, in_bytes,
                              rank * per, (rank + 1) * per)
        if field in OUTPUT_DIM:
            return slice_rows(raw, cfg["latent"], rank, degree)
        if field in INPUT_DIM:
            in_bytes = cfg["latent"] * 2
            per = in_bytes // degree
            return slice_cols(raw, len(raw) // in_bytes, in_bytes,
                              rank * per, (rank + 1) * per)
        if field in CONCAT_OUTPUT:
            half = len(raw) // 2
            return slice_rows(raw[:half], half and self._half_rows(name), rank,
                              degree) + \
                slice_rows(raw[half:], self._half_rows(name), rank, degree)
        if field in INPUT_DIM_PLAIN:
            rows = cfg["hidden"]
            in_bytes = len(raw) // rows
            per = in_bytes // degree
            if field == "shared_w2_weight" and \
                    (in_bytes // 2) % degree != 0:
                raise ShardFailure(f"{name}: input does not split {degree} ways")
            return slice_cols(raw, rows, in_bytes, rank * per, (rank + 1) * per)
        if field in ("expert_w1_weight", "expert_w1_scale"):
            return self._expert_gate_up(name, raw)
        if field in ("expert_w2_weight", "expert_w2_scale"):
            return self._expert_down(name, raw, field.endswith("scale"))
        raise ShardFailure(f"{name}: unclassified tensor, refusing to guess")

    def _half_rows(self, name):
        # shared_w1 is [2 * shared_inter, hidden]; dense_gate_up is
        # [2 * dense_inter, hidden] - rows of one half from the byte length
        raw = self.bytes_of(name)
        return (len(raw) // 2) // (self.config["hidden"] * 2)

    def _expert_gate_up(self, name, raw):
        cfg, degree, rank = self.config, self.degree, self.rank
        experts, inter = cfg["experts"], cfg["intermediate"]
        per_expert = len(raw) // experts
        half = per_expert // 2
        if inter % degree != 0:
            raise ShardFailure(f"{name}: intermediate over {degree} ranks")
        out = bytearray()
        for e in range(experts):
            block = raw[e * per_expert:(e + 1) * per_expert]
            out += slice_rows(block[:half], inter, rank, degree)
            out += slice_rows(block[half:], inter, rank, degree)
        return bytes(out)

    def _expert_down(self, name, raw, is_scale):
        cfg, degree, rank = self.config, self.degree, self.rank
        experts, inter, latent = cfg["experts"], cfg["intermediate"], cfg["latent"]
        k_per_rank = inter // degree
        if k_per_rank % GROUP != 0:
            raise ShardFailure(
                f"{name}: {k_per_rank} K elements per rank is not whole "
                f"MXFP4 groups; the scale plane cannot follow")
        per_expert = len(raw) // experts
        row_bytes = per_expert // latent
        per = (inter // GROUP if is_scale else inter // 2) // degree
        out = bytearray()
        for e in range(experts):
            block = raw[e * per_expert:(e + 1) * per_expert]
            out += slice_cols(block, latent, row_bytes,
                              rank * per, (rank + 1) * per)
        return bytes(out)


def main():
    if len(sys.argv) != 4:
        print("usage: k3_shard.py <in.pack> <out_prefix> <tp_degree>")
        return 2
    degree = int(sys.argv[3])
    if degree & (degree - 1) or degree == 0:
        print("SHARD FAILURE: tp_degree must be a power of two")
        return 1
    probe = Slicer(sys.argv[1], {}, degree, 0)
    cfg = probe.config
    needed = ("kda_heads", "kda_head", "heads", "kv_lora", "rope", "v_head")
    missing = [k for k in needed if k not in cfg]
    if missing:
        print(f"SHARD FAILURE: pack config lacks geometry {missing}; "
              f"repack with a current tools/k3_pack.py")
        return 1
    geo = {k: cfg[k] for k in needed}
    try:
        for rank in range(degree):
            slicer = Slicer(sys.argv[1], geo, degree, rank)
            tensors = slicer.emit(f"{sys.argv[2]}.rank{rank:02d}.pack")
        print(f"sharded {len(tensors)} tensors x {degree} ranks")
    except ShardFailure as failure:
        print(f"SHARD FAILURE: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
