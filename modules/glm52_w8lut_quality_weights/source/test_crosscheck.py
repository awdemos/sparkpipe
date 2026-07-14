import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import glm52_w8lut_codec as W


def main():
    dump = os.environ.get("W8LUT_DUMP")
    if not dump:
        raise SystemExit("W8LUT_DUMP is required")
    rng = np.random.default_rng(7)
    w = ((rng.standard_normal(300000) * 0.02).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    with tempfile.TemporaryDirectory(prefix="sparkpipe_w8lut_") as temp_dir:
        input_path = Path(temp_dir) / "input.bf16"
        output_path = Path(temp_dir) / "codes.bin"
        w.tofile(input_path)
        result = subprocess.run(
            [dump, str(input_path), str(output_path)],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(result.stderr or result.stdout)
        raw = output_path.read_bytes()
    c_e0 = int.from_bytes(raw[:2], "little")
    c_codes = np.frombuffer(raw[2:], np.uint8)
    codes, e0, stats = W.encode(w)
    if int(e0) != c_e0 or not np.array_equal(codes, c_codes):
        raise SystemExit("numpy != C")
    print("crosscheck PASS:", stats)


if __name__ == "__main__":
    main()
