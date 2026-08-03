#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "STATUS.md"


def main() -> int:
    text = STATUS.read_text(encoding="utf-8")
    required = (
        "HOST_BUILD_STATUS=SEE_EXTERNAL_VERIFICATION_RECEIPT",
        "HOST_TEST_STATUS=SEE_EXTERNAL_VERIFICATION_RECEIPT",
        "ARCHITECTURE_GATE_STATUS=SEE_EXTERNAL_VERIFICATION_RECEIPT",
        "CUDA13_SM121A_COMPILE_NOT_RUN=true",
        "BLACKWELL_EXECUTION_NOT_MEASURED=true",
        "PHYSICAL_NETWORK_EXECUTION_NOT_MEASURED=true",
        "PRODUCTION_READY=false",
    )
    forbidden = (
        "PRODUCTION_READY=true",
        "CUDA13_SM121A_COMPILE_VALIDATED=true",
        "BLACKWELL_EXECUTION_VALIDATED=true",
    )
    failures = []
    for marker in required:
        if marker not in text:
            failures.append(f"missing conservative status marker: {marker}")
    for marker in forbidden:
        if marker in text:
            failures.append(f"unsupported status claim: {marker}")
    if "external verification receipt" not in text.lower():
        failures.append("status does not bind claims to an external archive receipt")
    if failures:
        for failure in failures:
            print(f"  FAIL {failure}")
        return 1
    print("status remains conservative and receipt-bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
