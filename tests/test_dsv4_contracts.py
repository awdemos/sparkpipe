#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.run(
        ["python3", str(ROOT / "tools" / "generate_dsv4_contracts.py"), "--check"],
        check=True,
        cwd=ROOT,
    )
    flash = json.loads((ROOT / "model_contracts" / "dsv4_flash.json").read_text(encoding="utf-8"))
    pro = json.loads((ROOT / "model_contracts" / "dsv4_pro.json").read_text(encoding="utf-8"))
    assert flash["model"]["layer_count"] == 43
    assert pro["model"]["layer_count"] == 61
    assert len(flash["attention"]["compression_ratios"]) == 44
    assert len(pro["attention"]["compression_ratios"]) == 62
    assert flash["attention"]["compression_ratios"][:2] == [0, 0]
    assert pro["attention"]["compression_ratios"][:2] == [128, 128]
    assert flash["attention"]["compression_ratios"][-1] == 0
    assert pro["attention"]["compression_ratios"][-1] == 0
    assert flash["qualification"]["cuda_target"] == "sm_121a"
    assert pro["qualification"]["cuda_target"] == "sm_121a"
    print("PASS DSV4 Flash and Pro generated contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
