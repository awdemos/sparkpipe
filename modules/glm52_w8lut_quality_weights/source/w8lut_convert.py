#!/usr/bin/env python3
"""W8LUT v2 checkpoint converter: zai-org/GLM-5.2 (BF16 master) -> sparkpipe quality format.

Routing (see W8LUT_MIGRATION.md):
  layers 3..77 mlp.experts.*.{gate,up,down}_proj.weight  -> W8LUT v2 (u8 codes + u16 e0)
  everything else (trunk, MTP layer 78 incl. its experts) -> BF16 verbatim

Streams one shard at a time (peak disk ~5 GiB, ~10 GiB RAM), verifies every conversion, deletes as it goes.

NOTE: this emits generic safetensors (BF16 stored as U16 bit-patterns; safetensors-numpy
cannot express BF16). sparkpipe-native integration is a per-layer resident pack builder
patterned on tools/glm52_fp8_resident_pack.py (QUANT_MODE_W8LUT=3, e0 arrays in the
scale regions) using w8lut_np.encode as the conversion core -- see the migration doc.
  python3 w8lut_convert.py --out /data/glm52-w8lut [--shards 0:282] [--keep] [--dry]
"""
import argparse, json, os, re, struct, sys
import numpy as np
import w8lut_np as W

REPO = "zai-org/GLM-5.2"
EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$")


def is_w8lut_target(name):
    m = EXPERT.match(name)
    return m is not None and 3 <= int(m.group(1)) <= 77


def convert_shard(path, outdir, manifest):
    from safetensors.numpy import save_file
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n)); ds = 8 + n
    import mmap
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    out = {}
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        b, e = meta["data_offsets"]
        raw = np.frombuffer(mm[ds + b:ds + e], np.uint8)
        if meta["dtype"] == "BF16" and is_w8lut_target(name):
            w = raw.view(np.uint16)
            codes, e0, st = W.encode(w)
            dec = W.decode(codes, e0)
            r = W.rne4(w)
            ee = (r >> np.uint16(7)) & np.uint16(0xFF)
            inwin = (ee > e0) | ((ee == e0) & (((r >> np.uint16(3)) & np.uint16(0xF)) > 0))
            if not np.array_equal(dec[inwin], r[inwin]):
                raise RuntimeError(f"verify fail (in-window) {name}")
            out[name] = codes.reshape(meta["shape"])
            out[name + ".w8e0"] = np.array([e0], np.uint16)
            manifest.write(json.dumps({"name": name, "fmt": "w8lut2", "e0": int(e0),
                                       "below_ppm": st["below_ppm"], "shape": meta["shape"]}) + "\n")
        else:
            out[name] = raw.copy().view(np.uint16).reshape(-1) if meta["dtype"] == "BF16" else raw.copy()
            if meta["dtype"] == "BF16":
                out[name] = out[name].reshape(meta["shape"])
            manifest.write(json.dumps({"name": name, "fmt": meta["dtype"], "shape": meta["shape"]}) + "\n")
    mm.close(); f.close()
    save_file(out, os.path.join(outdir, "w8lut-" + os.path.basename(path)),
              metadata={"format": "w8lut2", "magic": "0x57384C32"})


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", default="0:282")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--work", default="_w8lut_work")
    args = ap.parse_args()
    a, b = (int(x) for x in args.shards.split(":"))
    os.makedirs(args.out, exist_ok=True)
    idx = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))[a:b]
    manifest = open(os.path.join(args.out, f"manifest_{a}_{b}.jsonl"), "w")
    for i, s in enumerate(shards):
        p = hf_hub_download(REPO, s, local_dir=args.work)
        convert_shard(p, args.out, manifest)
        manifest.flush()
        if not args.keep:
            os.remove(p)
        print(f"[{a + i + 1}/{a + len(shards)}] {s} done", flush=True)
    manifest.close()


if __name__ == "__main__":
    main()
