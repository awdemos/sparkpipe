"""One table, twice. The C classification table the loader will use
(model-families/k3/.../spark_k3_tp_shard_table.h) and the python slicer's
partition (tools/k3_shard.py) must classify every pack-manifest tensor
identically - a drift in either is a red suite, not a mis-sliced rank.

The gate packs the mini checkpoint, walks every manifest name, classifies it
through BOTH tables (the C one parsed structurally, the python one imported),
and compares. An unclassified name on either side fails: fail-closed is the
engine's contract and the tables must honour it in unison.
"""
import json
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
from test_k3_pack import mini_checkpoint  # noqa: E402
import k3_shard  # noqa: E402


def parse_c_table():
    text = (ROOT / "model-families/k3/include/sparkpipe/"
            "spark_k3_tp_shard_table.h").read_text()
    table = {}
    for block, klass in re.findall(
            r"if \(((?:.|\n)*?)\)\n\t\treturn (SPARK_TP_SHARD_CLASS_\w+);",
            text):
        for suffix in re.findall(r'"([^"]+)"', block):
            table[suffix] = klass
    return table


def python_class(field, name):
    if name in k3_shard.MODEL_REPLICATED or field in k3_shard.REPLICATED:
        return "SPARK_TP_SHARD_CLASS_REPLICATED"
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "SPARK_TP_SHARD_CLASS_OUTPUT_DIM"
    if field in k3_shard.OUTPUT_HEADS:
        return "SPARK_TP_SHARD_CLASS_OUTPUT_DIM_HEADS"
    if field in k3_shard.INPUT_HEADS:
        return "SPARK_TP_SHARD_CLASS_INPUT_DIM_HEADS"
    if field in k3_shard.OUTPUT_DIM:
        return "SPARK_TP_SHARD_CLASS_OUTPUT_DIM"
    if field in k3_shard.INPUT_DIM or field in k3_shard.INPUT_DIM_PLAIN or \
            field in ("expert_w2_weight", "expert_w2_scale"):
        return "SPARK_TP_SHARD_CLASS_INPUT_DIM"
    if field in k3_shard.CONCAT_OUTPUT or \
            field in ("expert_w1_weight", "expert_w1_scale"):
        return "SPARK_TP_SHARD_CLASS_CONCAT_OUTPUT"
    return "SPARK_TP_SHARD_CLASS_UNKNOWN"


def c_class(table, name):
    for suffix, klass in table.items():
        if name.endswith(suffix):
            return klass
    return "SPARK_TP_SHARD_CLASS_UNKNOWN"


def main():
    failures = 0
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        mini_checkpoint(root, latent=64, inter=64)
        pack = root / "mini.pack"
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_pack.py"),
                              str(root), str(pack)], capture_output=True, text=True)
        if run.returncode != 0:
            print("FAIL pack:", run.stdout[-200:])
            return 1
        raw = pack.read_bytes()
        _, _, length = struct.unpack_from("<IIQ", raw, 0)
        names = list(json.loads(raw[16:16 + length])["tensors"])
    table = parse_c_table()
    for name in names:
        field = name.split(".")[-1]
        c = c_class(table, name)
        p = python_class(field, name)
        if c != p:
            print(f"  FAIL {name}: C says {c}, python says {p}")
            failures += 1
        if c == "SPARK_TP_SHARD_CLASS_UNKNOWN":
            print(f"  FAIL {name}: unclassified in the C table")
            failures += 1
    print(f"names classified {len(names)}, C suffixes {len(table)}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\none table, twice - and both refuse what they do not know")
    return 0


if __name__ == "__main__":
    sys.exit(main())
