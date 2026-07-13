#!/usr/bin/env python3

import argparse
import importlib.util
import pathlib
import urllib.parse


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sparkpipe_api_stress", ROOT / "tools" / "sparkpipe_api_stress.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def arguments():
    return argparse.Namespace(
        model="glm-5.2",
        stream=True,
        temperature=0.0,
        max_completion_tokens=16,
    )


def main():
    chat = MODULE.build_request_body(
        arguments(),
        urllib.parse.urlparse("http://spark0:18080/v1/chat/completions"),
        "hello",
    )
    assert chat["messages"] == [{"role": "user", "content": "hello"}]
    assert chat["max_completion_tokens"] == 16
    assert "prompt" not in chat
    completion = MODULE.build_request_body(
        arguments(),
        urllib.parse.urlparse("http://spark0:18080/v1/completions"),
        "hello",
    )
    assert completion["prompt"] == "hello"
    assert completion["max_tokens"] == 16
    assert "messages" not in completion
    try:
        MODULE.build_request_body(
            arguments(), urllib.parse.urlparse("http://spark0:18080/other"), "hello"
        )
    except ValueError:
        return
    raise AssertionError("unsupported endpoint was accepted")


if __name__ == "__main__":
    main()
