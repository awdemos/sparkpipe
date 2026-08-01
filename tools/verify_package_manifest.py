#!/usr/bin/env python3
"""PACKAGE_MANIFEST.json must describe the tree as it is, not as it was.

The manifest is regenerated at packaging time, so between packagings every
source edit makes it stale - and nothing noticed, because nothing checked.
This gate compares the manifest against the tracked tree: every listed path
must exist with the recorded sha256, and every tracked packaged file must be
listed. It reports drift; it does not fix it.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "PACKAGE_MANIFEST.json"
# tracked files the package deliberately does not carry: the packaging
# metadata itself, retained validation logs, and qualification eval data
NOT_PACKAGED = {"PACKAGE_MANIFEST.json", "SHA256SUMS"}
NOT_PACKAGED_PREFIXES = ("docs/validation-logs/", "qualification/")


def tracked_files():
    listing = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    for path in listing:
        if not (ROOT / path).is_file():
            continue
        if path in NOT_PACKAGED:
            continue
        if any(path.startswith(prefix) for prefix in NOT_PACKAGED_PREFIXES):
            continue
        yield path


def main():
    manifest = json.loads(MANIFEST.read_text())
    entries = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    failures = 0

    if manifest.get("file_count") != len(manifest["files"]):
        print(f"  FAIL file_count is {manifest.get('file_count')}, "
              f"files list holds {len(manifest['files'])}")
        failures += 1

    for path, recorded in entries.items():
        target = ROOT / path
        if not target.is_file():
            print(f"  FAIL {path}: listed but missing from the tree")
            failures += 1
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != recorded:
            print(f"  FAIL {path}: sha256 drifted")
            failures += 1

    packaged = list(tracked_files())
    unlisted = [p for p in packaged if p not in entries]
    for path in unlisted:
        print(f"  FAIL {path}: in the tree but not in the manifest")
        failures += 1

    print(f"manifest entries {len(entries)}, packaged tree files "
          f"{len(packaged)}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nmanifest matches the tree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
