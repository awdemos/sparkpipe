#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "build" / "test_glm52_prompt_pipeline_input"
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()
    command = [
        "python3",
        str(root / "tools" / "glm52_prompt_pipeline_input.py"),
        "--token-ids",
        "101,202,303",
        "--output-dir",
        str(output_dir),
        "--pipeline-output-dir",
        str(output_dir / "pipeline"),
    ]
    completed = subprocess.run(command, cwd=str(root), text=True, capture_output=True, check=True)
    assert "glm52_prompt_bootstrap_token=303" in completed.stdout
    token_json = output_dir / "prompt_tokens.json"
    env_file = output_dir / "pipeline_env.sh"
    payload = json.loads(token_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "sparkpipe.glm52.prompt_pipeline_input.v1"
    assert payload["token_ids"] == [101, 202, 303]
    assert payload["bootstrap_token_id"] == 303
    assert payload["pipeline_semantics"] == "single-token bootstrap only; no prompt KV prefill context yet"
    env_text = env_file.read_text(encoding="utf-8")
    assert "GLM52_LOCAL_PIPELINE_INPUT_TOKEN_ID=303" in env_text
    assert "GLM52_PROMPT_TOKEN_COUNT=3" in env_text


if __name__ == "__main__":
    main()
