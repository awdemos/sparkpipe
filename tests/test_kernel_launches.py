#!/usr/bin/env python3
"""A kernel launch must match the kernel it launches.

Three uint32_t parameters in a row all type-check in any order. LmMoeFinalizeKernel
takes (tokens, top_k, dimension) and kimi_k3 passed (dimension, top_k, tokens),
with the expert id where the packed row belongs, under a 1D grid for a kernel that
reads blockIdx.y. Four defects, one compile, no warning.

Two things are checkable without understanding the code:

  A kernel that reads blockIdx.y needs a 2D grid. If the launch's first <<< >>>
  argument is not a dim3, blockIdx.y is zero and only the first slice of the
  work is done. This is mechanical and has no false positives.

  A kernel parameter named for a count should not receive an expression named
  for a width, and vice versa. This is a heuristic on names, so it only fires
  when the mismatch is unambiguous - an argument that looks like a dimension
  arriving where the parameter is called tokens or rows.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_DIRS = ["inference/kernels", "runtime"]
CALLER_GLOBS = ["inference/llms"]

COUNT_WORDS = ("tokens", "rows", "count", "heads", "layers")
WIDTH_WORDS = ("dimension", "width", "hidden", "_dim")


def kernel_signatures():
    """name -> (parameter names, whether the body reads blockIdx.y)."""
    found = {}
    for directory in KERNEL_DIRS:
        base = os.path.join(ROOT, directory)
        for root, _, files in os.walk(base):
            for name in files:
                if not name.endswith((".cuh", ".h")):
                    continue
                text = open(os.path.join(root, name), errors="replace").read()
                for match in re.finditer(
                        r"void (Lm\w+Kernel)\(([^)]*)\)\s*\{", text, re.S):
                    body_start = match.end()
                    depth, index = 1, body_start
                    while index < len(text) and depth:
                        if text[index] == "{":
                            depth += 1
                        elif text[index] == "}":
                            depth -= 1
                        index += 1
                    parameters = [p.strip().split()[-1].lstrip("*")
                                  for p in match.group(2).split(",") if p.strip()]
                    found[match.group(1)] = (
                        parameters, "blockIdx.y" in text[body_start:index])
    return found


def call_sites():
    sites = []
    for directory in CALLER_GLOBS:
        for root, _, files in os.walk(os.path.join(ROOT, directory)):
            for name in files:
                if not name.endswith((".cu", ".cuh")):
                    continue
                path = os.path.join(root, name)
                text = re.sub(r"//[^\n]*", "", open(path, errors="replace").read())
                flat = re.sub(r"\s+", " ", text)
                for match in re.finditer(
                        r"(Lm\w+Kernel)\s*<[^>]*>\s*<<<(.*?)>>>\s*\((.*?)\);", flat):
                    sites.append((os.path.relpath(path, ROOT), match.group(1),
                                  match.group(2), match.group(3)))
    return sites


def split_arguments(text):
    out, depth, current = [], 0, ""
    for character in text:
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        if character == "," and depth == 0:
            out.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        out.append(current.strip())
    return out


def main():
    kernels = kernel_signatures()
    sites = call_sites()
    failures = 0
    checked_grid = checked_names = 0
    for path, name, grid, arguments in sites:
        if name not in kernels:
            continue
        parameters, uses_y = kernels[name]
        if uses_y:
            checked_grid += 1
            if not grid.strip().startswith("dim3"):
                print(f"  FAIL {path}: {name} reads blockIdx.y and is launched "
                      f"with a 1D grid")
                failures += 1
        values = split_arguments(arguments)
        if len(values) != len(parameters):
            continue
        for parameter, value in zip(parameters, values):
            lowered = value.lower()
            if any(w in parameter for w in COUNT_WORDS) and \
                    any(w in lowered for w in WIDTH_WORDS) and \
                    not any(w in lowered for w in COUNT_WORDS):
                checked_names += 1
                print(f"  FAIL {path}: {name}({parameter}=...) receives "
                      f"'{value}', which names a width")
                failures += 1
    print(f"kernels {len(kernels)}  launches {len(sites)}  "
          f"blockIdx.y grids checked {checked_grid}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nevery launch matches the kernel's grid use and parameter names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
