"""The shard geometry reassembles to the pack, byte for byte.

The mini checkpoint packs, then slices at TP 2, and every class is put back
together by its own rule and compared to the original bytes: replicated
tensors equal on both ranks, head-block output splits concatenate, input
splits interleave column-wise, the concatenated gate|up tensors reassemble
half by half per expert, and expert w2's packed-nibble K split rebuilds with
its scale plane. Then TP 4 on the same mini must be REFUSED - two heads do
not split four ways and sixteen K elements are not a whole MXFP4 group - and
the refusal must be loud, not a mis-sliced pack.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from test_k3_pack import mini_checkpoint  # noqa: E402


def read_pack(path):
    raw = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<IIQ", raw, 0)
    assert magic == 0x4B33504B
    manifest = json.loads(raw[16:16 + length])
    base = 16 + length
    base += (-base) % 64

    def tensor(name):
        entry = manifest["tensors"][name]
        return raw[base + entry["offset"]: base + entry["offset"] + entry["bytes"]]
    return manifest, tensor


def join_cols(parts, rows):
    per = [len(p) // rows for p in parts]
    out = bytearray()
    for r in range(rows):
        for part, width in zip(parts, per):
            out += part[r * width:(r + 1) * width]
    return bytes(out)


def main():
    failures = 0
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        mini_checkpoint(root, latent=64, inter=64)
        pack = root / "mini.pack"
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_pack.py"),
                              str(root), str(pack)], capture_output=True, text=True)
        if run.returncode != 0:
            print("FAIL pack:", run.stdout[-300:])
            return 1
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_shard.py"),
                              str(pack), str(root / "mini"), "2"],
                             capture_output=True, text=True)
        if run.returncode != 0:
            print("FAIL shard:", run.stdout[-300:])
            return 1
        full_manifest, full = read_pack(pack)
        ranks = [read_pack(root / f"mini.rank{r:02d}.pack") for r in range(2)]
        cfg = full_manifest["config"]

        def both(name):
            return [ranks[r][1](name) for r in range(2)]

        # replicated: equal on both ranks and equal to the source
        for name in ("model.norm.weight", "model.layers.0.attn_norm_weight",
                     "model.layers.1.mla_kv_a_weight",
                     "model.layers.0.router_weight"):
            a, b = both(name)
            if not (a == b == full(name)):
                print(f"  FAIL {name}: replication is not replication")
                failures += 1
        # output rows concatenate: vocab shard and a head-block shard
        for name in ("model.embed_tokens.weight",
                     "model.layers.0.kda_q_weight",
                     "model.layers.1.mla_q_up_weight",
                     "model.layers.0.routed_down_weight"):
            if b"".join(both(name)) != full(name):
                print(f"  FAIL {name}: output shards do not reassemble")
                failures += 1
        # input columns interleave: the all-reduce-closed projections
        for name, rows in (("model.layers.0.kda_out_weight", cfg["hidden"]),
                           ("model.layers.1.mla_out_weight", cfg["hidden"]),
                           ("model.layers.0.routed_up_weight", cfg["hidden"]),
                           ("model.layers.0.shared_w2_weight", cfg["hidden"])):
            if join_cols(both(name), rows) != full(name):
                print(f"  FAIL {name}: input shards do not reassemble")
                failures += 1
        # concatenated gate|up: per half, and per expert for w1
        name = "model.layers.0.shared_w1_weight"
        a, b = both(name)
        half = len(a) // 2
        rebuilt = a[:half] + b[:half] + a[half:] + b[half:]
        if rebuilt != full(name):
            print("  FAIL shared gate|up halves do not reassemble gate-first")
            failures += 1
        for name, per_rank_row in (("model.layers.0.expert_w1_weight",
                                    cfg["latent"] // 2),
                                   ("model.layers.0.expert_w1_scale",
                                    cfg["latent"] // 32)):
            a, b = both(name)
            experts, inter = cfg["experts"], cfg["intermediate"]
            block = len(a) // experts
            half = block // 2
            rebuilt = bytearray()
            for e in range(experts):
                ea, eb = a[e * block:(e + 1) * block], b[e * block:(e + 1) * block]
                rebuilt += ea[:half] + eb[:half] + ea[half:] + eb[half:]
            if bytes(rebuilt) != full(name):
                print(f"  FAIL {name}: expert gate|up shards do not reassemble")
                failures += 1
        for name in ("model.layers.0.expert_w2_weight",
                     "model.layers.0.expert_w2_scale"):
            a, b = both(name)
            experts, latent = cfg["experts"], cfg["latent"]
            block_a = len(a) // experts
            rebuilt = bytearray()
            for e in range(experts):
                rebuilt += join_cols([a[e * block_a:(e + 1) * block_a],
                                      b[e * block_a:(e + 1) * block_a]], latent)
            if bytes(rebuilt) != full(name):
                print(f"  FAIL {name}: the packed-K shards do not reassemble")
                failures += 1
        # TP 4 must refuse: two heads, and sixteen K elements per rank
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_shard.py"),
                              str(pack), str(root / "bad"), "4"],
                             capture_output=True, text=True)
        if run.returncode == 0 or "FAILURE" not in run.stdout:
            print("  FAIL a misaligned degree was not refused")
            failures += 1
    print(f"tensors sharded {len(full_manifest['tensors'])} x 2 ranks")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nevery class reassembles to the pack, and a degree the "
          "geometry cannot honour is refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
