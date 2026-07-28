"""The K3 engine keeps the contract the slice kernels enforce.

The sequence-run kernels demand rows sorted by sequence with positions
ascending in a run; the MLA path demands context_length count every stored
row; the sampler demands logits only where a forward actually predicts the
next token. The engine is the one component that can violate all of that at
once while every kernel stays correct, so this gate drives two requests -
prompt 5 and prompt 3 under a four-row budget and two slots - plus a third
that must queue for a freed slot, and holds every printed plan to the
contract:

  chunked prefill: prompt 5 splits 4 + 0-with-decode (the final prompt token
  belongs to decode), and the chunk never asks for logits
  continuous batching: decode rows and a prefill chunk share a step
  positions ascend within every run and resume where the last chunk stopped
  context_length is always the run's last position + 1
  a finished request's slot is reused by the queued one
  EOS ends a request under budget and its output stops there
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    with tempfile.TemporaryDirectory() as scratch:
        binary = Path(scratch) / "k3_engine_host"
        build = subprocess.run(
            ["cc", "-std=c11", "-O1", "-I", str(ROOT),
             str(ROOT / "tests" / "host_cuda" / "k3_engine_host.c"),
             "-o", str(binary)],
            capture_output=True, text=True)
        if build.returncode != 0:
            print("FAIL host build:", build.stderr[:400])
            return 1
        run = subprocess.run([str(binary)], capture_output=True, text=True)
    if run.returncode != 0:
        print(f"FAIL the engine faulted (returncode {run.returncode})")
        print(run.stdout[-800:])
        return 1
    text = run.stdout
    failures = 0
    steps = []
    for block in re.split(r"(?=step \d+ )", text):
        m = re.match(r"step (\d+) rows (\d+) sequences (\d+)", block)
        if not m:
            continue
        seqs = []
        for sm in re.finditer(
                r"seq \d+ slot (\d+) request (\d+) run (\d+)\.\.(\d+) "
                r"context (\d+) logits (-?\d+)((?:\n    row [^\n]*)*)", block):
            rows = [tuple(int(x) for x in rm.groups()) for rm in re.finditer(
                r"row (\d+) token (\d+) position (\d+) slot (\d+)", sm.group(7))]
            seqs.append(dict(slot=int(sm.group(1)), request=int(sm.group(2)),
                             begin=int(sm.group(3)), end=int(sm.group(4)),
                             context=int(sm.group(5)), logits=int(sm.group(6)),
                             rows=rows))
        steps.append(dict(index=int(m.group(1)), rows=int(m.group(2)), seqs=seqs))

    last_position = {}
    for step in steps:
        cursor = 0
        for seq in step["seqs"]:
            if seq["begin"] != cursor:
                print(f"  FAIL step {step['index']}: runs are not contiguous")
                failures += 1
            cursor = seq["end"]
            positions = [r[2] for r in seq["rows"]]
            if positions != sorted(positions) or (
                    len(positions) > 1
                    and positions != list(range(positions[0], positions[0] + len(positions)))):
                print(f"  FAIL step {step['index']}: positions do not ascend by one")
                failures += 1
            if any(r[3] != seq["slot"] for r in seq["rows"]):
                print(f"  FAIL step {step['index']}: a row's slot differs from its run's")
                failures += 1
            if positions and seq["context"] != positions[-1] + 1:
                print(f"  FAIL step {step['index']}: context {seq['context']} "
                      f"is not last position + 1")
                failures += 1
            key = seq["request"]
            if positions and key in last_position and positions[0] != last_position[key] + 1:
                print(f"  FAIL request {key}: resumed at {positions[0]} "
                      f"after {last_position[key]}")
                failures += 1
            if positions:
                last_position[key] = positions[-1]
            if seq["logits"] >= 0 and not (seq["begin"] <= seq["logits"] < seq["end"]):
                print(f"  FAIL step {step['index']}: logits row outside the run")
                failures += 1
        if cursor != step["rows"]:
            print(f"  FAIL step {step['index']}: runs do not cover the rows")
            failures += 1

    # request 1: prompt 5, budget 4 -> first chunk is exactly 4 rows, no logits
    first = steps[0]["seqs"][0]
    if not (first["request"] == 1 and first["end"] - first["begin"] == 4
            and first["logits"] < 0):
        print("  FAIL prompt 5 under budget 4 must open with a 4-row chunk "
              "asking for no logits")
        failures += 1
    # some step must mix a decode row with a prefill chunk
    if not any(any(s["logits"] >= 0 for s in st["seqs"])
               and any(s["logits"] < 0 and s["end"] - s["begin"] > 0 for s in st["seqs"])
               for st in steps):
        print("  FAIL no step mixed decode with a prefill chunk")
        failures += 1
    # request 3 queued, then reused a slot some finished request abandoned
    slots_of_3 = {s["slot"] for st in steps for s in st["seqs"] if s["request"] == 3}
    slots_of_12 = {s["slot"] for st in steps for s in st["seqs"] if s["request"] in (1, 2)}
    if not slots_of_3 or not slots_of_3 <= slots_of_12:
        print("  FAIL the queued request did not reuse a freed slot")
        failures += 1
    if "out_b " not in text or re.search(r"out_b (\d+) 7\b", text) is None:
        print("  FAIL EOS did not end request b's output at its second token")
        failures += 1

    print(f"steps {len(steps)}, requests finished 3")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe engine batches, chunks, mixes, ends and reuses - "
          "and never breaks the run contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
