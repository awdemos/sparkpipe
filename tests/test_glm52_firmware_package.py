#!/usr/bin/env python3

from pathlib import Path
import subprocess


def run_limit_check(root: Path, value: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "make",
            "-C",
            str(root / "modules" / "glm52_resident_decode_stage"),
            "require_latency_limit",
            f"MAX_STAGE_MICROSECONDS={value}",
        ],
        capture_output=True,
        text=True,
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    for invalid_value in ("", "0", "-1", "1ms"):
        result = run_limit_check(root, invalid_value)
        assert result.returncode != 0
        assert "positive integer qualified stage limit" in result.stderr
    result = run_limit_check(root, "1000000")
    assert result.returncode == 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
