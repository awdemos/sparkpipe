"""The metric, enforced: non-test code size only shrinks. Tests are exempt
(they are the Solutions-protection); everything else - model code included -
is codesize. A rise fails loudly with the delta; lowering the ceiling is
part of landing a shrink, exactly like the naming law's budgets. The
philosophy: the smaller the codebase, the more of it fits in one head (or
one context window) at once, and reasoning quality follows."""
import subprocess, sys
from pathlib import Path

CEILING = 104792  # lines; only goes down

ROOT = Path(__file__).resolve().parent.parent
EXTS = {'.c', '.h', '.cu', '.cuh', '.py', '.mk', '.sh'}

def main():
    total = 0
    for p in ROOT.rglob('*'):
        rel = p.relative_to(ROOT)
        if not p.is_file() or rel.parts[0] in ('tests', '.git', 'docs'):
            continue
        if p.suffix in EXTS or p.name == 'Makefile':
            total += sum(1 for _ in p.open(errors='surrogateescape'))
    print(f"non-test lines: {total} (ceiling {CEILING})")
    if total > CEILING:
        print(f"\nFAIL codesize grew by {total - CEILING} over the ceiling; "
              f"shrink it or justify a new ceiling in the same commit")
        return 1
    if total < CEILING - 800:
        print(f"note: ceiling is {CEILING - total} above reality; "
              f"lower it with your next landing")
    print("\nthe codebase did not grow")
    return 0

if __name__ == '__main__':
    sys.exit(main())
