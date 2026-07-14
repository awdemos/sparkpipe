#!/usr/bin/env python3

import hashlib
import json
import pathlib
import subprocess
import tempfile


def main():
    repository = pathlib.Path(__file__).resolve().parents[1]
    tool = repository / "tools" / "sparkpipe_release_assemble.py"
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        template = root / "template"
        output = root / "output"
        diagnostic_output = root / "diagnostic-output"
        plain_output = root / "plain-output"
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
            "files": [{"path": "bin/runtime", "bytes": 3, "sha256": "0" * 64}],
            "roles": [
                {
                    "name": "pp13_rank_daemon",
                    "argv": ["--max-active", "16"],
                    "env": ["KEEP_RANK=1"],
                },
                {
                    "name": "pp13_cuda_residentd",
                    "argv": ["--max-active", "16"],
                    "env": ["KEEP_RESIDENT=1", "SPARKPIPE_PP13_TRACE=1",
                            "SPARKPIPE_MTP_GPU_PROFILE=1"],
                },
                {
                    "name": "spark0_gateway",
                    "argv": ["--max-active", "16"],
                    "env": ["KEEP_GATEWAY=1"],
                },
            ],
        }
        (template / "sparkpipe.json").write_text(json.dumps(manifest),encoding="utf-8")
        subprocess.run([
            "python3",str(tool),
            "--template",str(template),
            "--output",str(diagnostic_output),
            "--release-id","diagnostic",
            "--git-commit","abc123",
            "--max-active","64",
            "--kv-pool-tokens","65536",
            "--kv-logical-blocks","1024",
            "--mtp",
            "--replace","bin/runtime=" + str(replacement),
        ],check=True)
        diagnostic = json.loads(
            (diagnostic_output / "sparkpipe.json").read_text(encoding="utf-8"))
        diagnostic_by_role = {role["name"]: role for role in diagnostic["roles"]}
        assert "SPARKPIPE_PP13_TRACE=1" in diagnostic_by_role[
            "spark0_gateway"]["env"]
        assert "SPARKPIPE_STAGE_COMPLETION_DEBUG=1" in diagnostic_by_role[
            "pp13_cuda_residentd"]["env"]
        assert "SPARKPIPE_STAGE_PHASE_HASH=1" in diagnostic_by_role[
            "pp13_cuda_residentd"]["env"]
        assert "SPARKPIPE_HIDDEN_DUMP_DIR={state_root}/hidden_dumps" in (
            diagnostic_by_role["pp13_cuda_residentd"]["env"])
        assert "SPARKPIPE_PP13_TRACE=1" not in diagnostic_by_role[
            "pp13_cuda_residentd"]["env"]
        assert "SPARKPIPE_STAGE_COMPLETION_DEBUG=1" in diagnostic_by_role[
            "pp13_rank_daemon"]["env"]
        assert "SPARKPIPE_PP13_TRACE=1" in diagnostic_by_role[
            "pp13_rank_daemon"]["env"]
        subprocess.run([
            "python3",str(tool),
            "--template",str(template),
            "--output",str(output),
            "--release-id","new",
            "--git-commit","abc123",
            "--max-active","64",
            "--kv-pool-tokens","65536",
            "--kv-logical-blocks","1024",
            "--mtp",
            "--without-diagnostics",
            "--role-env-unset","pp13_cuda_residentd=SPARKPIPE_MTP_GPU_PROFILE",
            "--role-env","spark0_gateway=KEEP_GATEWAY=2",
            "--replace","bin/runtime=" + str(replacement),
        ],check=True)
        result = json.loads((output / "sparkpipe.json").read_text(encoding="utf-8"))
        expected = hashlib.sha256(b"new-runtime").hexdigest()
        assert result["release_id"] == "new"
        assert result["git_commit"] == "abc123"
        assert result["max_active_sequence_count"] == 64
        assert result["roles"][0]["argv"] == ["--max-active","64"]
        assert result["roles"][1]["argv"] == [
            "--max-active","64","--kv-pool-tokens","65536","--mtp"]
        assert result["roles"][2]["argv"] == [
            "--max-active","64","--kv-logical-blocks","1024","--mtp"]
        for role in result["roles"]:
            assert "SPARKPIPE_RELEASE_ID=new" in role["env"]
            assert "SPARKPIPE_RELEASE_GIT_COMMIT=abc123" in role["env"]
            assert any(item.startswith("SPARKPIPE_RELEASE_GENERATION=")
                       for item in role["env"])
        assert "KEEP_RANK=1" in result["roles"][0]["env"]
        assert "KEEP_RESIDENT=1" in result["roles"][1]["env"]
        assert "SPARKPIPE_MTP_GPU_PROFILE=1" not in result["roles"][1]["env"]
        assert "KEEP_GATEWAY=2" in result["roles"][2]["env"]
        assert "KEEP_GATEWAY=1" not in result["roles"][2]["env"]
        diagnostic_names = {
            "SPARKPIPE_STAGE_COMPLETION_DEBUG",
            "SPARKPIPE_STAGE_PHASE_HASH",
            "SPARKPIPE_HIDDEN_DUMP_DIR",
            "SPARKPIPE_PP13_TRACE",
        }
        assert all(entry.split("=",1)[0] not in diagnostic_names
                   for role in result["roles"] for entry in role["env"])
        assert result["files"][0]["bytes"] == 11
        assert result["files"][0]["sha256"] == expected
        assert (output / "bin" / "runtime").read_bytes() == b"new-runtime"
        assert list(root.glob("output.assembling.*")) == []
        missing_mode = subprocess.run([
            "python3",str(tool),
            "--template",str(template),
            "--output",str(root / "missing-mode"),
            "--release-id","missing-mode",
            "--git-commit","abc123",
            "--kv-logical-blocks","1024",
        ],capture_output=True,text=True)
        assert missing_mode.returncode != 0
        assert "one of the arguments --mtp --plain-decode is required" in (
            missing_mode.stderr)
        subprocess.run([
            "python3",str(tool),
            "--template",str(diagnostic_output),
            "--output",str(plain_output),
            "--release-id","plain",
            "--git-commit","abc123",
            "--kv-logical-blocks","1024",
            "--plain-decode",
            "--without-diagnostics",
        ],check=True)
        plain = json.loads(
            (plain_output / "sparkpipe.json").read_text(encoding="utf-8"))
        assert all("--mtp" not in role["argv"] for role in plain["roles"])


if __name__ == "__main__":
    main()
