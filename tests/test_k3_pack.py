"""The packer moves the checkpoint's bytes and performs exactly the folds the
weight table cannot get from any checkpoint.

A miniature synthetic checkpoint - written with the same stdlib safetensors
framing the reader parses, no torch - drives tools/k3_pack.py end to end, and
the manifest's bytes are held to the source:

  the expert payloads and E8M0 planes are BIT-PRESERVED, gate first then up,
  experts-major, the layout the weight-only GEMM prices
  the q-fold equals the einsum it claims, to bf16 round-to-nearest-even
  kv_b's value half lands as its own per-head table
  the attention-residual gammas fold elementwise into their score rows
  an E8M0 0xff anywhere is a loud failure, as is a missing tensor
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(7)


def bf16(shape):
    return (rng.integers(0, 1 << 16, size=shape, dtype=np.uint16)
            & np.uint16(0x7FBF)).astype(np.uint16)  # finite-ish, no NaN class


def write_safetensors(path, tensors):
    header, offset = {}, 0
    blobs = []
    for name, (dtype, array) in tensors.items():
        raw = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)


def mini_checkpoint(root, poison_scale=False, latent=32, inter=32):
    hidden, vocab = 32, 64
    q_lora, kv_lora, rope, nope, v_head, heads = 16, 32, 8, 16, 32, 2
    kda_heads, kda_head = 2, 32
    kda_dim = kda_heads * kda_head
    experts = 4
    config = {"hidden_size": hidden, "num_hidden_layers": 3, "vocab_size": vocab,
              "num_experts": experts, "num_experts_per_tok": 2,
              "routed_expert_hidden_size": latent, "moe_intermediate_size": inter,
              "num_shared_experts": 1, "q_lora_rank": q_lora,
              "kv_lora_rank": kv_lora, "qk_rope_head_dim": rope,
              "qk_nope_head_dim": nope, "v_head_dim": v_head,
              "num_attention_heads": heads,
              "linear_attn_config": {"num_heads": kda_heads, "head_dim": kda_head,
                                     "short_conv_kernel_size": 4},
              "layer_types": ["linear_attention", "full_attention",
                              "linear_attention"]}
    (root / "config.json").write_text(json.dumps(config))
    t = {}
    t["model.embed_tokens.weight"] = ("BF16", bf16((vocab, hidden)))
    t["model.norm.weight"] = ("BF16", bf16((hidden,)))
    t["lm_head.weight"] = ("BF16", bf16((vocab, hidden)))
    t["model.output_attn_res_norm.weight"] = ("BF16", bf16((hidden,)))
    t["model.output_attn_res_proj.weight"] = ("BF16", bf16((1, hidden)))
    for layer, kind in enumerate(config["layer_types"]):
        p = f"model.layers.{layer}."
        a, m = p + "self_attn.", p + "mlp."
        t[p + "input_layernorm.weight"] = ("BF16", bf16((hidden,)))
        t[p + "post_attention_layernorm.weight"] = ("BF16", bf16((hidden,)))
        for res in ("self_attention_res", "mlp_res"):
            t[p + res + "_norm.weight"] = ("BF16", bf16((hidden,)))
            t[p + res + "_proj.weight"] = ("BF16", bf16((1, hidden)))
        if kind == "linear_attention":
            for proj in "qkv":
                t[a + proj + "_proj.weight"] = ("BF16", bf16((kda_dim, hidden)))
                t[a + proj + "_conv1d.weight"] = ("BF16", bf16((kda_dim, 1, 4)))
            t[a + "f_a_proj.weight"] = ("BF16", bf16((kda_head, hidden)))
            t[a + "f_b_proj.weight"] = ("BF16", bf16((kda_dim, kda_head)))
            t[a + "dt_bias"] = ("F32", rng.standard_normal(kda_dim).astype(np.float32))
            t[a + "A_log"] = ("F32", rng.standard_normal(kda_heads).astype(np.float32))
            t[a + "b_proj.weight"] = ("BF16", bf16((kda_heads, hidden)))
            t[a + "g_a_proj.weight"] = ("BF16", bf16((kda_head, hidden)))
            t[a + "g_b_proj.weight"] = ("BF16", bf16((kda_dim, kda_head)))
            t[a + "o_norm.weight"] = ("BF16", bf16((kda_head,)))
            t[a + "o_proj.weight"] = ("BF16", bf16((hidden, kda_dim)))
        else:
            t[a + "q_a_proj.weight"] = ("BF16", bf16((q_lora, hidden)))
            t[a + "q_a_layernorm.weight"] = ("BF16", bf16((q_lora,)))
            t[a + "q_b_proj.weight"] = ("BF16", bf16((heads * (nope + rope), q_lora)))
            t[a + "kv_a_proj_with_mqa.weight"] = ("BF16", bf16((kv_lora + rope, hidden)))
            t[a + "kv_a_layernorm.weight"] = ("BF16", bf16((kv_lora,)))
            t[a + "kv_b_proj.weight"] = ("BF16", bf16((heads * (nope + v_head), kv_lora)))
            t[a + "g_a_proj.weight"] = ("BF16", bf16((v_head, hidden)))
            t[a + "g_b_proj.weight"] = ("BF16", bf16((heads * v_head, v_head)))
            t[a + "o_proj.weight"] = ("BF16", bf16((hidden, heads * v_head)))
        t[m + "gate.weight"] = ("BF16", bf16((experts, hidden)))
        t[m + "gate.e_score_correction_bias"] = (
            "F32", rng.standard_normal(experts).astype(np.float32))
        t[m + "routed_expert_down_proj.weight"] = ("BF16", bf16((latent, hidden)))
        t[m + "routed_expert_up_proj.weight"] = ("BF16", bf16((hidden, latent)))
        t[m + "routed_expert_norm.weight"] = ("BF16", bf16((latent,)))
        t[m + "shared_experts.gate_proj.weight"] = ("BF16", bf16((inter, hidden)))
        t[m + "shared_experts.up_proj.weight"] = ("BF16", bf16((inter, hidden)))
        t[m + "shared_experts.down_proj.weight"] = ("BF16", bf16((hidden, inter)))
        for e in range(experts):
            base = m + f"experts.{e}."
            for name, rows, cols in (("w1", inter, latent), ("w3", inter, latent),
                                     ("w2", latent, inter)):
                t[base + name + ".weight"] = (
                    "U8", rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8))
                scale = rng.integers(100, 150, (rows, cols // 32), dtype=np.uint8)
                if poison_scale and layer == 2 and e == 3 and name == "w2":
                    scale[0, 0] = 0xFF
                t[base + name + ".weight_scale"] = ("U8", scale)
    write_safetensors(root / "model.safetensors", t)
    return t


def read_pack(path):
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from("<IIQ", raw, 0)
    assert magic == 0x4B33504B and version == 1
    manifest = json.loads(raw[16:16 + length])
    base = 16 + length
    base += (-base) % 64
    def tensor(name):
        entry = manifest["tensors"][name]
        return raw[base + entry["offset"]: base + entry["offset"] + entry["bytes"]]
    return manifest, tensor


def main():
    failures = 0
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        src = mini_checkpoint(root)
        out = root / "mini.pack"
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_pack.py"),
                              str(root), str(out)], capture_output=True, text=True)
        if run.returncode != 0:
            print("FAIL packer:", (run.stdout + run.stderr)[-400:])
            return 1
        manifest, tensor = read_pack(out)
        p = "model.layers.0."
        want = b"".join(
            src[f"{p}mlp.experts.{e}.{w}.weight"][1].tobytes()
            for e in range(4) for w in ("w1", "w3"))
        if tensor(p + "expert_w1_weight") != want:
            print("  FAIL expert w1|w3 payload is not bit-preserved gate-first")
            failures += 1
        want = b"".join(
            src[f"{p}mlp.experts.{e}.{w}.weight_scale"][1].tobytes()
            for e in range(4) for w in ("w1", "w3"))
        if tensor(p + "expert_w1_scale") != want:
            print("  FAIL the E8M0 plane is not [expert][neuron][k_group]")
            failures += 1
        # q-fold on layer 1
        p = "model.layers.1."
        heads, nope, rope, v_head, kv_lora, q_lora = 2, 16, 8, 32, 32, 16
        def f32(u): return (u.astype(np.uint32) << 16).view(np.float32)
        q_b = f32(src[p + "self_attn.q_b_proj.weight"][1]).reshape(
            heads, nope + rope, q_lora)
        kv_b = f32(src[p + "self_attn.kv_b_proj.weight"][1]).reshape(
            heads, nope + v_head, kv_lora)
        absorbed = np.einsum("hnl,hnq->hlq", kv_b[:, :nope], q_b[:, :nope])
        folded = np.concatenate([absorbed, q_b[:, nope:]], axis=1)
        u = folded.astype(np.float32).view(np.uint32)
        want = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        got = np.frombuffer(tensor(p + "mla_q_up_weight"), np.uint16)
        if not np.array_equal(got, want.reshape(-1)):
            print("  FAIL the q-fold does not equal its einsum")
            failures += 1
        value = kv_b[:, nope:].reshape(heads * v_head, kv_lora)
        u = value.view(np.uint32)
        want = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).reshape(-1)
        if not np.array_equal(
                np.frombuffer(tensor(p + "mla_kv_b_value_weight"), np.uint16), want):
            print("  FAIL kv_b's value half is not its own per-head table")
            failures += 1
        np.seterr(over="ignore")
        gamma = f32(src[p + "self_attention_res_norm.weight"][1])
        proj = f32(src[p + "self_attention_res_proj.weight"][1])
        u = (proj * gamma).view(np.uint32)
        want = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).reshape(-1)
        if not np.array_equal(
                np.frombuffer(tensor(p + "attnres_attn_weight"), np.uint16), want):
            print("  FAIL the res-norm gamma did not fold into the score rows")
            failures += 1
        # a poisoned scale must be a loud failure, not a packed NaN
        for name in list(root.iterdir()):
            if name.suffix == ".pack" or name.name.endswith(".payload"):
                name.unlink()
        mini_checkpoint(root, poison_scale=True)
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_pack.py"),
                              str(root), str(out)], capture_output=True, text=True)
        if run.returncode == 0 or "0xff" not in run.stdout:
            print("  FAIL an E8M0 0xff was not refused")
            failures += 1
    print(f"tensors packed {len(manifest['tensors'])}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe pack is the checkpoint's bytes, plus exactly the folds "
          "the table owes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
