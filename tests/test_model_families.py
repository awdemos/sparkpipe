#!/usr/bin/env python3
# The family conformance gate: every model family the engine claims must
# have its firmware config, its host geometry header, and its stage
# doorway - and where host and firmware both state a dimension, the two
# must agree. A family missing a piece fails here before it fails at
# link time on someone else's afternoon.
import os, re, sys

FAMILIES = {
    "glm52":  {"llms": "inference/llms/glm5_2",      "doorway": "modules/glm52_resident_decode_stage",
               "host": "model-families/glm52/include/sparkpipe/spark_glm52_model.h",
               "pairs": []},
    "k3":     {"llms": "inference/llms/kimi_k3",     "doorway": "modules/k3_resident_decode_stage",
               "host": "model-families/k3/include/sparkpipe/spark_k3_kv_geometry.h",
               "pairs": []},
    "mimo25": {"llms": "inference/llms/mimo_2_5",    "doorway": "modules/mimo25_resident_decode_stage",
               "host": "model-families/mimo25/include/sparkpipe/spark_mimo25_model.h",
               "pairs": []},
    "qwen36": {"llms": "inference/llms/qwen_3_6",    "doorway": "modules/qwen36_resident_decode_stage",
               "host": "model-families/qwen36/include/sparkpipe/spark_qwen36_model.h",
               "pairs": [("QWEN36_HIDDEN", "SPARK_QWEN36_MODEL_HIDDEN_DIMENSION"),
                          ("QWEN36_LAYERS", "SPARK_QWEN36_MODEL_LAYER_COUNT"),
                          ("QWEN36_VOCAB", "SPARK_QWEN36_MODEL_VOCAB_COUNT"),
                          ("QWEN36_HEAD_DIM", "SPARK_QWEN36_MODEL_HEAD_DIMENSION")]},
    "dsv4":   {"llms": "inference/llms/deepseek_v4", "doorway": "modules/dsv4_resident_decode_stage",
               "host": "model-families/dsv4/include/sparkpipe/spark_dsv4_model.h",
               "pairs": [("DSV4_HIDDEN", "SPARK_DSV4_MODEL_HIDDEN_DIMENSION"),
                          ("DSV4_LAYERS", "SPARK_DSV4_MODEL_LAYER_COUNT"),
                          ("DSV4_VOCAB", "SPARK_DSV4_MODEL_VOCAB_COUNT"),
                          ("DSV4_HEAD_DIM", "SPARK_DSV4_MODEL_HEAD_DIMENSION")]},
}

def defines(path):
    text = open(path, errors="surrogateescape").read()
    return dict(re.findall(r"#define (\w+) (\d+)u", text))

bad = 0
for family, spec in FAMILIES.items():
    config = os.path.join(spec["llms"], "config.h")
    doorway = None
    if os.path.isdir(spec["doorway"]):
        for root, _, files in os.walk(spec["doorway"]):
            for name in files:
                if name.endswith("validation.c"):
                    doorway = os.path.join(root, name)
    for label, path in [("firmware config", config), ("host geometry", spec["host"]), ("stage doorway", doorway or "")]:
        if not path or not os.path.isfile(path):
            print(f"FAIL {family}: missing {label} ({path or spec['doorway']})")
            bad += 1
    if spec["pairs"] and os.path.isfile(config) and os.path.isfile(spec["host"]):
        firmware, host = defines(config), defines(spec["host"])
        for fw_key, host_key in spec["pairs"]:
            if firmware.get(fw_key) != host.get(host_key):
                print(f"FAIL {family}: {fw_key}={firmware.get(fw_key)} vs {host_key}={host.get(host_key)}")
                bad += 1
print(f"{len(FAMILIES)} families checked, {bad} problems")
sys.exit(1 if bad else 0)
