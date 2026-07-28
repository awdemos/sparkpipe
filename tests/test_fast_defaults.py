"""Fast is the default; slow is a named kill-switch; every mode announces
itself. A disabled speed booster is a fallback wearing a configuration's
clothes, and this gate keeps the tree from growing one back.

Structural assertions on the sources - CPU-provable, hardware-independent:
the backend takes the scheduler's full default flags and clears prefix
reuse ONLY under its kill-switch; the release path awaits the resident
ONLY under its kill-switch; the gateway enables DSpark by default with
--no-dspark and the env to turn it off; and the effective-configuration
banner exists so no mode is silent. Each kill-switch must both exist and
be the ONLY route to the slow path.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    failures = 0
    backend = (ROOT / "node/backend.c").read_text()
    gateway = (ROOT / "api/gateway/http_server.c").read_text()

    clears = re.findall(
        r"~SPARK_SCHEDULER_CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE",
        backend)
    guarded = re.search(
        r'getenv\("SPARKPIPE_DISABLE_PREFIX_REUSE"\)[^}]*?'
        r'~SPARK_SCHEDULER_CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE',
        backend, re.S)
    if len(clears) != 1 or guarded is None:
        print("  FAIL prefix reuse is cleared outside its kill-switch "
              f"({len(clears)} clears)")
        failures += 1

    release = re.search(
        r"SubmitReleaseToResident\((?:.|\n)*?^\}", backend, re.M)
    if release is None or \
            'getenv("SPARKPIPE_RELEASE_SYNC_AWAIT") == 0' not in release.group(0) or \
            release.group(0).index("SPARKPIPE_RELEASE_SYNC_AWAIT") > \
            release.group(0).index("AwaitSubmitResult"):
        print("  FAIL the release await is not gated behind its kill-switch")
        failures += 1

    if 'getenv("SPARKPIPE_DISABLE_DSPARK") == 0 ? 1u : 0u' not in gateway or \
            '"--no-dspark"' not in gateway:
        print("  FAIL DSpark is not default-on with a named kill-switch")
        failures += 1

    if "ring_effective_config" not in backend:
        print("  FAIL the effective-configuration banner is missing")
        failures += 1
    else:
        # the format string is one literal split across source lines
        start = backend.index("ring_effective_config")
        banner = backend[start:backend.index(");", start)]
        for mode in ("prefix_reuse=", "release=", "dspark=", "mtp="):
            if mode not in banner:
                print(f"  FAIL the banner does not announce {mode}")
                failures += 1

    print("kill-switches audited 3, banner modes 4")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nfast is the default, slow is named, and nothing is silent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
