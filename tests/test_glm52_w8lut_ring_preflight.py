#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def load_tool(repository: Path):
    path = repository / "tools" / "glm52_w8lut_ring_preflight.py"
    specification = importlib.util.spec_from_file_location(
        "glm52_w8lut_ring_preflight",
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load W8LUT ring preflight")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    tool = load_tool(repository)
    commit = "a" * 40
    arguments = argparse.Namespace(
        checkout_root="/home/{host}/checkout",
        stagepack_root="/artifacts/{rank}/stage",
        w8lut_pack_root="/artifacts/{rank}/w8",
        connect_timeout=1,
        command_timeout=1,
        expected_commit=commit,
        one_layer_per_rank=True,
        sample_bytes=256,
        verify_sha256=False,
    )
    assert tool.selected_layer(0) == 3
    assert tool.selected_layer(2) == 12
    assert tool.selected_layer(12) == 72
    assert tool.parse_ranks([2, 0]) == [0, 2]
    try:
        tool.parse_ranks([2, 2])
    except tool.RingPreflightFailure:
        pass
    else:
        raise AssertionError("duplicate ranks were accepted")

    def successful_remote(host, remote_arguments, connect_timeout, command_timeout):
        del host, connect_timeout, command_timeout
        if remote_arguments[:3] == ["git", "-C", "/home/spark2/checkout"]:
            if remote_arguments[3:] == ["rev-parse", "HEAD"]:
                output = commit + "\n"
            else:
                output = ""
        else:
            assert remote_arguments[-2:] == ["--layer", "12"]
            output = json.dumps({"status": "ok", "rank": 2}) + "\n"
        return subprocess.CompletedProcess(remote_arguments, 0, output, "")

    tool.run_remote = successful_remote
    result = tool.validate_rank(2, "spark2", arguments)
    assert result["status"] == "ok"
    assert result["rank"] == 2
    assert result["git_commit"] == commit

    def wrong_commit(host, remote_arguments, connect_timeout, command_timeout):
        del host, connect_timeout, command_timeout
        return subprocess.CompletedProcess(
            remote_arguments,
            0,
            ("b" * 40) + "\n",
            "",
        )

    tool.run_remote = wrong_commit
    result = tool.validate_rank(2, "spark2", arguments)
    assert result["status"] == "failed"
    assert "checkout commit" in result["error"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
