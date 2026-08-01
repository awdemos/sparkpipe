#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def role_by_name(manifest: dict, role_name: str) -> dict:
    matches = [role for role in manifest["roles"] if role["name"] == role_name]
    assert len(matches) == 1
    return matches[0]


def argument_value(role: dict, argument: str) -> str:
    arguments = role["argv"]
    matches = [index for index, value in enumerate(arguments) if value == argument]
    assert len(matches) == 1
    index = matches[0]
    assert index + 1 < len(arguments)
    return arguments[index + 1]


def run_tool(tool: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(tool), *arguments],
        check=check,
        capture_output=not check,
        text=True,
    )


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    tool = repository / "tools" / "sparkpipe_release_assemble.py"

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        template = root / "template"
        replacement = root / "replacement.bin"
        (template / "bin").mkdir(parents=True)
        (template / "bin" / "runtime").write_bytes(b"old")
        replacement.write_bytes(b"new-runtime")
        manifest = {
            "schema_version": 1,
            "release_id": "old",
            "generation": 1,
            "git_commit": "old",
            "max_active_sequence_count": 16,
            "files": [
                {
                    "path": "bin/runtime",
                    "bytes": 3,
                    "sha256": "0" * 64,
                }
            ],
            "roles": [
                {
                    "name": "ring_rank_daemon",
                    "argv": ["--max-active", "16"],
                    "env": ["KEEP_RANK=1"],
                },
                {
                    "name": "ring_cuda_residentd",
                    "argv": [
                        "--max-active",
                        "16",
                        "--fp8-pack-root",
                        "/packs",
                        "--stagepack-root",
                        "/packs",
                    ],
                    "env": [
                        "KEEP_RESIDENT=1",
                        "SPARKPIPE_RING_TRACE=1",
                        "SPARKPIPE_MTP_GPU_PROFILE=1",
                    ],
                },
                {
                    "name": "spark0_gateway",
                    "argv": [
                        "--max-active",
                        "16",
                        "--decode-batch-target",
                        "13",
                        "--fp8-pack-root",
                        "/packs",
                        "--stagepack-root",
                        "/packs",
                    ],
                    "env": ["KEEP_GATEWAY=1"],
                },
            ],
        }
        (template / "sparkpipe.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        diagnostic_output = root / "diagnostic-output"
        run_tool(
            tool,
            "--template",
            str(template),
            "--output",
            str(diagnostic_output),
            "--release-id",
            "diagnostic",
            "--git-commit",
            "abc123",
            "--max-active",
            "64",
            "--kv-pool-tokens",
            "65536",
            "--kv-logical-blocks",
            "1024",
            "--mtp",
            "--dspark",
            "--dspark-model-dir",
            "/models/dspark",
            "--dspark-manifest",
            "/models/dspark/dspark_manifest.json",
            "--replace",
            "bin/runtime=" + str(replacement),
        )
        diagnostic = json.loads(
            (diagnostic_output / "sparkpipe.json").read_text(encoding="utf-8")
        )
        gateway = role_by_name(diagnostic, "spark0_gateway")
        resident = role_by_name(diagnostic, "ring_cuda_residentd")
        rank = role_by_name(diagnostic, "ring_rank_daemon")
        assert "SPARKPIPE_RING_TRACE=1" in gateway["env"]
        assert "SPARKPIPE_STAGE_COMPLETION_DEBUG=1" in resident["env"]
        assert "SPARKPIPE_STAGE_PHASE_HASH=1" in resident["env"]
        assert (
            "SPARKPIPE_HIDDEN_DUMP_DIR={state_root}/hidden_dumps"
            in resident["env"]
        )
        assert "SPARKPIPE_RING_TRACE=1" not in resident["env"]
        assert "SPARKPIPE_STAGE_COMPLETION_DEBUG=1" in rank["env"]
        assert "SPARKPIPE_RING_TRACE=1" in rank["env"]
        assert "--dspark" in gateway["argv"]
        assert "--dspark" in resident["argv"]
        assert argument_value(resident, "--dspark-config") == "/models/dspark/config.json"
        assert argument_value(resident, "--dspark-manifest") == (
            "/models/dspark/dspark_manifest.json"
        )
        assert argument_value(resident, "--dspark-safetensors") == (
            "/models/dspark/model.safetensors"
        )
        assert argument_value(resident, "--model-quantization") == "fp8"
        assert argument_value(gateway, "--model-quantization") == "fp8"
        assert argument_value(resident, "--stagepack-root") == "/packs"
        assert argument_value(resident, "--moe-pack-root") == "/packs"
        assert all(
            "--fp8-pack-root" not in role["argv"]
            for role in diagnostic["roles"]
        )

        output = root / "output"
        run_tool(
            tool,
            "--template",
            str(template),
            "--output",
            str(output),
            "--release-id",
            "new",
            "--git-commit",
            "abc123",
            "--max-active",
            "64",
            "--kv-pool-tokens",
            "65536",
            "--kv-logical-blocks",
            "1024",
            "--mtp",
            "--without-diagnostics",
            "--role-env-unset",
            "ring_cuda_residentd=SPARKPIPE_MTP_GPU_PROFILE",
            "--role-env",
            "spark0_gateway=KEEP_GATEWAY=2",
            "--replace",
            "bin/runtime=" + str(replacement),
        )
        result = json.loads(
            (output / "sparkpipe.json").read_text(encoding="utf-8")
        )
        assert result["release_id"] == "new"
        assert result["git_commit"] == "abc123"
        assert result["max_active_sequence_count"] == 64
        assert result["files"][0]["bytes"] == len(b"new-runtime")
        assert result["files"][0]["sha256"] == hashlib.sha256(
            b"new-runtime"
        ).hexdigest()
        assert (output / "bin" / "runtime").read_bytes() == b"new-runtime"
        assert list(root.glob("output.assembling.*")) == []
        for role in result["roles"]:
            assert "SPARKPIPE_RELEASE_ID=new" in role["env"]
            assert "SPARKPIPE_RELEASE_GIT_COMMIT=abc123" in role["env"]
            assert any(
                entry.startswith("SPARKPIPE_RELEASE_GENERATION=")
                for entry in role["env"]
            )
        result_gateway = role_by_name(result, "spark0_gateway")
        result_resident = role_by_name(result, "ring_cuda_residentd")
        assert "KEEP_GATEWAY=2" in result_gateway["env"]
        assert "KEEP_GATEWAY=1" not in result_gateway["env"]
        assert "SPARKPIPE_MTP_GPU_PROFILE=1" not in result_resident["env"]
        diagnostic_names = {
            "SPARKPIPE_STAGE_COMPLETION_DEBUG",
            "SPARKPIPE_STAGE_PHASE_HASH",
            "SPARKPIPE_HIDDEN_DUMP_DIR",
            "SPARKPIPE_RING_TRACE",
        }
        assert all(
            entry.split("=", 1)[0] not in diagnostic_names
            for role in result["roles"]
            for entry in role["env"]
        )

        plain_output = root / "plain-output"
        run_tool(
            tool,
            "--template",
            str(diagnostic_output),
            "--output",
            str(plain_output),
            "--release-id",
            "plain",
            "--git-commit",
            "abc123",
            "--kv-logical-blocks",
            "1024",
            "--plain-decode",
            "--without-diagnostics",
        )
        plain = json.loads(
            (plain_output / "sparkpipe.json").read_text(encoding="utf-8")
        )
        assert all("--mtp" not in role["argv"] for role in plain["roles"])
        assert all("--dspark" not in role["argv"] for role in plain["roles"])

        nvfp4_output = root / "nvfp4-output"
        run_tool(
            tool,
            "--template",
            str(template),
            "--output",
            str(nvfp4_output),
            "--release-id",
            "nvfp4",
            "--git-commit",
            "abc123",
            "--kv-logical-blocks",
            "1024",
            "--model-quantization",
            "nvfp4",
            "--stagepack-root",
            "/home/{host}/artifacts/nvfp4-stage",
            "--moe-pack-root",
            "/home/{host}/artifacts/nvfp4-moe",
            "--mtp",
        )
        nvfp4 = json.loads(
            (nvfp4_output / "sparkpipe.json").read_text(encoding="utf-8")
        )
        for role_name in ("spark0_gateway", "ring_cuda_residentd"):
            role = role_by_name(nvfp4, role_name)
            assert argument_value(role, "--model-quantization") == "nvfp4"
            assert argument_value(role, "--stagepack-root") == (
                "/home/{host}/artifacts/nvfp4-stage"
            )
            assert argument_value(role, "--moe-pack-root") == (
                "/home/{host}/artifacts/nvfp4-moe"
            )

        missing_mode = run_tool(
            tool,
            "--template",
            str(template),
            "--output",
            str(root / "missing-mode"),
            "--release-id",
            "missing-mode",
            "--git-commit",
            "abc123",
            "--kv-logical-blocks",
            "1024",
            check=False,
        )
        assert missing_mode.returncode != 0
        assert "one of the arguments --mtp --plain-decode is required" in (
            missing_mode.stderr
        )

        missing_nvfp4_stage = run_tool(
            tool,
            "--template",
            str(template),
            "--output",
            str(root / "missing-nvfp4-stage"),
            "--release-id",
            "missing-nvfp4-stage",
            "--git-commit",
            "abc123",
            "--kv-logical-blocks",
            "1024",
            "--model-quantization",
            "nvfp4",
            "--moe-pack-root",
            "/nvfp4-moe",
            "--mtp",
            check=False,
        )
        assert missing_nvfp4_stage.returncode != 0
        assert "--stagepack-root is required" in missing_nvfp4_stage.stderr
        assert list(root.glob("missing-nvfp4-stage.assembling.*")) == []

        missing_dspark_artifacts = run_tool(
            tool,
            "--template",
            str(template),
            "--output",
            str(root / "missing-dspark-artifacts"),
            "--release-id",
            "missing-dspark-artifacts",
            "--git-commit",
            "abc123",
            "--kv-logical-blocks",
            "1024",
            "--plain-decode",
            "--dspark",
            check=False,
        )
        assert missing_dspark_artifacts.returncode != 0
        assert "--dspark-model-dir is required" in missing_dspark_artifacts.stderr
        assert list(root.glob("missing-dspark-artifacts.assembling.*")) == []

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
