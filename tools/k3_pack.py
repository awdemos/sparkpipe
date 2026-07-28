#!/usr/bin/env python3
"""Pack a Kimi K3 checkpoint into the sparkpipe resident layout.

The engines serve the checkpoint as shipped: MXFP4 expert payloads and their
E8M0 scales are moved, never recomputed, and everything on the ignore list
stays BF16. What the packer OWES beyond moving bytes is exactly what the
layer's weight table cannot get from any checkpoint directly:

  q-fold        kv_b's k_nope half absorbed into q_b, per head:
                A[h] = kv_b_k[h]^T @ q_b_nope[h], then the 64 unrotated rows -
                heads * (kv_lora + rope) rows over q_lora
  kv_b split    the value half of kv_b as its own [heads * v_head, kv_lora]
                per-head table (the gate lives in v-space and does not
                commute, so o_proj stays as shipped)
  w1 | w3       each routed expert's gate and up projections concatenated
                gate-first - the SiTU kernel's contract - payload and scales
                alike, experts-major for the rank-3 descriptor
  scale plane   E8M0 bytes relaid to [expert][neuron][k_group], the layout
                LmGemmWeightOnlyConsume prices
  gamma folds   every attention-residual score projection multiplied
                elementwise by its RMSNorm gamma, because the kernel norms
                without a weight and the fold is exact
  conv flatten  [channels, 1, kernel] to [channels, kernel]

Output is one file: magic, version, a JSON manifest of {tensor name ->
offset, bytes} keyed by the K3LayerWeights field names (model.layers.N.
prefixed), and 64-byte-aligned payload. A loader walks names to pointers and
needs to know nothing else.

Torch is not required. BF16 rides as uint16; the two folds run in float32
through numpy and round to nearest even on the way back. Failure is loud:
a missing tensor, an E8M0 0xff, a group size that is not 32, an expert count
that is not the config's - each is a PackFailure naming what and where.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x4B33504B  # 'K3PK'
VERSION = 1
ALIGN = 64
E8M0_NAN = 0xFF
GROUP = 32


class PackFailure(RuntimeError):
    pass


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    u = f32.astype(np.float32).view(np.uint32)
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


DTYPE_BYTES = {"BF16": 2, "F32": 4, "U8": 1, "I64": 8, "F16": 2}


class SafetensorDir:
    """Minimal reader: index.json plus shards, or a single model.safetensors.
    Returns raw bytes and the header's shape/dtype; interpretation is the
    caller's."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        index = self.model_dir / "model.safetensors.index.json"
        if index.is_file():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            single = self.model_dir / "model.safetensors"
            if not single.is_file():
                raise PackFailure(f"no safetensors index or file in {model_dir}")
            self.weight_map = None
            self.single = single.name
        self.headers = {}

    def _header(self, shard):
        if shard not in self.headers:
            path = self.model_dir / shard
            with open(path, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(length))
            header.pop("__metadata__", None)
            self.headers[shard] = (header, 8 + length)
        return self.headers[shard]

    def names(self):
        if self.weight_map is not None:
            return set(self.weight_map)
        return set(self._header(self.single)[0])

    def tensor(self, name):
        shard = self.weight_map.get(name) if self.weight_map is not None \
            else (self.single if name in self._header(self.single)[0] else None)
        if shard is None:
            raise PackFailure(f"missing tensor: {name}")
        header, base = self._header(shard)
        entry = header[name]
        begin, end = entry["data_offsets"]
        with open(self.model_dir / shard, "rb") as handle:
            handle.seek(base + begin)
            raw = handle.read(end - begin)
        return entry["dtype"], tuple(entry["shape"]), raw

    def bf16(self, name, shape=None):
        dtype, got, raw = self.tensor(name)
        if dtype != "BF16":
            raise PackFailure(f"{name}: expected BF16, checkpoint says {dtype}")
        if shape is not None and got != tuple(shape):
            raise PackFailure(f"{name}: shape {got}, expected {tuple(shape)}")
        return np.frombuffer(raw, dtype=np.uint16).reshape(got)

    def u8(self, name, shape=None):
        dtype, got, raw = self.tensor(name)
        if dtype != "U8":
            raise PackFailure(f"{name}: expected U8, checkpoint says {dtype}")
        if shape is not None and got != tuple(shape):
            raise PackFailure(f"{name}: shape {got}, expected {tuple(shape)}")
        return np.frombuffer(raw, dtype=np.uint8).reshape(got)


def quant_pair(reader, base):
    """The routed experts as compressed-tensors serialises them. Two spellings
    exist in the wild; both are probed and anything else is loud."""
    names = reader.names()
    if base + ".weight_packed" in names:
        return base + ".weight_packed", base + ".weight_scale"
    if base + ".weight" in names and base + ".weight_scale" in names:
        return base + ".weight", base + ".weight_scale"
    raise PackFailure(f"{base}: no recognised quantised serialisation "
                      f"(.weight_packed or .weight + .weight_scale)")


def check_scales(name, scales):
    if int((scales == E8M0_NAN).sum()) != 0:
        raise PackFailure(f"{name}: E8M0 0xff (NaN) in the scale plane")


class Pack:
    def __init__(self, out_path):
        self.handle = open(out_path, "wb")
        self.manifest = {}
        self.offset = 0

    def add(self, name, payload):
        pad = (-self.offset) % ALIGN
        if pad:
            self.handle.write(b"\0" * pad)
            self.offset += pad
        raw = payload.tobytes()
        self.manifest[name] = {"offset": self.offset, "bytes": len(raw)}
        self.handle.write(raw)
        self.offset += len(raw)



def pack_model(model_dir, out_path):
    reader = SafetensorDir(model_dir)
    config = json.loads((Path(model_dir) / "config.json").read_text())
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    experts = config["num_experts"]
    top_k = config["num_experts_per_tok"]
    latent = config["routed_expert_hidden_size"]
    inter = config["moe_intermediate_size"]
    shared = config.get("num_shared_experts", 1) * inter
    q_lora = config["q_lora_rank"]
    kv_lora = config["kv_lora_rank"]
    rope = config["qk_rope_head_dim"]
    nope = config["qk_nope_head_dim"]
    v_head = config["v_head_dim"]
    heads = config["num_attention_heads"]
    kda_heads = config["linear_attn_config"]["num_heads"] \
        if "linear_attn_config" in config else config["num_attention_heads"]
    kda_head = config["linear_attn_config"]["head_dim"] \
        if "linear_attn_config" in config else config["head_dim"]
    kda_dim = kda_heads * kda_head
    kernel = config["linear_attn_config"]["short_conv_kernel_size"] \
        if "linear_attn_config" in config else 4
    types = config["layer_types"]
    if len(types) != layers:
        raise PackFailure("layer_types does not cover num_hidden_layers")
    if latent % GROUP != 0 or inter % GROUP != 0:
        raise PackFailure("expert dims are not whole MXFP4 groups")

    payload_path = Path(str(out_path) + ".payload")
    pack = Pack(payload_path)
    L = "model.layers.{}."

    def bf(dst, src, shape=None):
        pack.add(dst, reader.bf16(src, shape))

    # model level
    bf("model.embed_tokens.weight", "model.embed_tokens.weight",
       (config["vocab_size"], hidden))
    bf("model.norm.weight", "model.norm.weight", (hidden,))
    bf("lm_head.weight", "lm_head.weight", (config["vocab_size"], hidden))
    gamma = bf16_to_f32(reader.bf16("model.output_attn_res_norm.weight", (hidden,)))
    proj = bf16_to_f32(reader.bf16("model.output_attn_res_proj.weight", (1, hidden)))
    pack.add("model.attnres_out_weight", f32_to_bf16(proj * gamma))

    for layer in range(layers):
        p = L.format(layer)
        linear = types[layer] == "linear_attention"
        bf(p + "attn_norm_weight", p + "input_layernorm.weight", (hidden,))
        bf(p + "mlp_norm_weight", p + "post_attention_layernorm.weight", (hidden,))
        for side, norm in (("attnres_attn_weight", "self_attention_res"),
                           ("attnres_mlp_weight", "mlp_res")):
            g = bf16_to_f32(reader.bf16(p + f"{norm}_norm.weight", (hidden,)))
            w = bf16_to_f32(reader.bf16(p + f"{norm}_proj.weight", (1, hidden)))
            pack.add(p + side, f32_to_bf16(w * g))
        if linear:
            a = p + "self_attn."
            bf(p + "kda_q_weight", a + "q_proj.weight", (kda_dim, hidden))
            bf(p + "kda_k_weight", a + "k_proj.weight", (kda_dim, hidden))
            bf(p + "kda_v_weight", a + "v_proj.weight", (kda_dim, hidden))
            for conv in "qkv":
                w = reader.bf16(a + f"{conv}_conv1d.weight", (kda_dim, 1, kernel))
                pack.add(p + f"kda_{conv}_conv_weight", w.reshape(kda_dim, kernel))
            bf(p + "kda_decay_down_weight", a + "f_a_proj.weight", (kda_head, hidden))
            bf(p + "kda_decay_up_weight", a + "f_b_proj.weight", (kda_dim, kda_head))
            dtype, shape, raw = reader.tensor(a + "dt_bias")
            pack.add(p + "kda_decay_bias", np.frombuffer(raw, np.uint16 if dtype == "BF16" else np.float32))
            dtype, shape, raw = reader.tensor(a + "A_log")
            pack.add(p + "kda_head_log_scale", np.frombuffer(raw, np.uint16 if dtype == "BF16" else np.float32))
            bf(p + "kda_beta_weight", a + "b_proj.weight", (kda_heads, hidden))
            bf(p + "kda_gate_down_weight", a + "g_a_proj.weight", (kda_head, hidden))
            bf(p + "kda_gate_up_weight", a + "g_b_proj.weight", (kda_dim, kda_head))
            bf(p + "kda_out_norm_weight", a + "o_norm.weight", (kda_head,))
            bf(p + "kda_out_weight", a + "o_proj.weight", (hidden, kda_dim))
        else:
            a = p + "self_attn."
            bf(p + "mla_q_down_weight", a + "q_a_proj.weight", (q_lora, hidden))
            bf(p + "mla_q_norm_weight", a + "q_a_layernorm.weight", (q_lora,))
            bf(p + "mla_kv_a_weight", a + "kv_a_proj_with_mqa.weight",
               (kv_lora + rope, hidden))
            bf(p + "mla_kv_a_norm_weight", a + "kv_a_layernorm.weight", (kv_lora,))
            q_b = bf16_to_f32(reader.bf16(a + "q_b_proj.weight",
                                          (heads * (nope + rope), q_lora)))
            kv_b = bf16_to_f32(reader.bf16(a + "kv_b_proj.weight",
                                           (heads * (nope + v_head), kv_lora)))
            q_b = q_b.reshape(heads, nope + rope, q_lora)
            kv_b = kv_b.reshape(heads, nope + v_head, kv_lora)
            absorbed = np.einsum("hnl,hnq->hlq", kv_b[:, :nope, :], q_b[:, :nope, :])
            folded = np.concatenate([absorbed, q_b[:, nope:, :]], axis=1)
            pack.add(p + "mla_q_up_weight",
                     f32_to_bf16(folded.reshape(heads * (kv_lora + rope), q_lora)))
            pack.add(p + "mla_kv_b_value_weight",
                     f32_to_bf16(kv_b[:, nope:, :].reshape(heads * v_head, kv_lora)))
            bf(p + "mla_gate_down_weight", a + "g_a_proj.weight", (v_head, hidden))
            bf(p + "mla_gate_up_weight", a + "g_b_proj.weight", (heads * v_head, v_head))
            bf(p + "mla_out_weight", a + "o_proj.weight", (hidden, heads * v_head))
        m = p + "mlp."
        if m + "gate.weight" not in reader.names():
            # the dense layer: one MLP, no router, no experts
            w1 = reader.bf16(m + "gate_proj.weight")
            w3 = reader.bf16(m + "up_proj.weight")
            pack.add(p + "dense_gate_up_weight", np.concatenate([w1, w3], axis=0))
            bf(p + "dense_down_weight", m + "down_proj.weight")
            continue
        bf(p + "router_weight", m + "gate.weight", (experts, hidden))
        dtype, shape, raw = reader.tensor(m + "gate.e_score_correction_bias")
        pack.add(p + "router_bias", np.frombuffer(raw, np.float32 if dtype == "F32" else np.uint16))
        bf(p + "routed_down_weight", m + "routed_expert_down_proj.weight",
           (latent, hidden))
        bf(p + "routed_up_weight", m + "routed_expert_up_proj.weight",
           (hidden, latent))
        bf(p + "routed_norm_weight", m + "routed_expert_norm.weight", (latent,))
        s1 = reader.bf16(m + "shared_experts.gate_proj.weight", (shared, hidden))
        s3 = reader.bf16(m + "shared_experts.up_proj.weight", (shared, hidden))
        pack.add(p + "shared_w1_weight", np.concatenate([s1, s3], axis=0))
        bf(p + "shared_w2_weight", m + "shared_experts.down_proj.weight",
           (hidden, shared))
        w1_pay, w1_sc, w2_pay, w2_sc = [], [], [], []
        for e in range(experts):
            base = m + f"experts.{e}."
            g_name, g_scale = quant_pair(reader, base + "w1")
            u_name, u_scale = quant_pair(reader, base + "w3")
            d_name, d_scale = quant_pair(reader, base + "w2")
            g = reader.u8(g_name, (inter, latent // 2))
            u = reader.u8(u_name, (inter, latent // 2))
            d = reader.u8(d_name, (latent, inter // 2))
            gs = reader.u8(g_scale, (inter, latent // GROUP))
            us = reader.u8(u_scale, (inter, latent // GROUP))
            ds = reader.u8(d_scale, (latent, inter // GROUP))
            for name, sc in ((g_scale, gs), (u_scale, us), (d_scale, ds)):
                check_scales(name, sc)
            w1_pay.append(np.concatenate([g, u], axis=0))
            w1_sc.append(np.concatenate([gs, us], axis=0))
            w2_pay.append(d)
            w2_sc.append(ds)
        pack.add(p + "expert_w1_weight", np.stack(w1_pay))
        pack.add(p + "expert_w1_scale", np.stack(w1_sc))
        pack.add(p + "expert_w2_weight", np.stack(w2_pay))
        pack.add(p + "expert_w2_scale", np.stack(w2_sc))

    pack.handle.flush()
    pack.handle.close()
    echo = {"hidden": hidden, "layers": layers, "experts": experts,
            "top_k": top_k, "latent": latent, "intermediate": inter,
            "group": GROUP, "vocab": config["vocab_size"],
            "kda_heads": kda_heads, "kda_head": kda_head, "heads": heads,
            "kv_lora": kv_lora, "rope": rope, "v_head": v_head,
            "nope": nope, "shared": shared, "q_lora": q_lora}
    manifest = json.dumps({"config": echo, "tensors": pack.manifest},
                          separators=(",", ":")).encode()
    with open(out_path, "wb") as out:
        out.write(struct.pack("<IIQ", MAGIC, VERSION, len(manifest)))
        out.write(manifest)
        pad = (-out.tell()) % ALIGN
        out.write(b"\0" * pad)
        with open(payload_path, "rb") as body:
            while True:
                chunk = body.read(1 << 24)
                if not chunk:
                    break
                out.write(chunk)
    payload_path.unlink()
    return echo, pack.manifest


def main():
    if len(sys.argv) != 3:
        print("usage: k3_pack.py <checkpoint_dir> <out.pack>")
        return 2
    try:
        echo, manifest = pack_model(sys.argv[1], sys.argv[2])
    except PackFailure as failure:
        print(f"PACK FAILURE: {failure}")
        return 1
    print(f"packed {len(manifest)} tensors, "
          f"{sum(t['bytes'] for t in manifest.values())} payload bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
