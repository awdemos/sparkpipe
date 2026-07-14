#!/usr/bin/env python3
"""G4 harness: logit-KL of a candidate weight format vs the BF16 reference.

Protocol: run the SAME prompt set (>= 2M tokens, fixed seed, greedy teacher-forced) through
sparkpipe twice -- once with weights_format=bf16ref, once with the candidate -- dumping
per-token logits (or top-256 logits + logsumexp) to .npz shards. Then:
  python3 kl_eval.py ref_dir cand_dir
Acceptance (W8LUT v2): mean KL(ref || cand) <= 0.5 x mean KL(ref || fp8blk baseline),
top-1 agreement >= fp8blk baseline. Integration of the dump hook is sparkpipe-side (TODO).
"""
import sys, glob
import numpy as np


def kl_from_pair(ref_logits, cand_logits):
    r = ref_logits.astype(np.float64); c = cand_logits.astype(np.float64)
    r = r - r.max(-1, keepdims=True); c = c - c.max(-1, keepdims=True)
    pr = np.exp(r); pr /= pr.sum(-1, keepdims=True)
    lr = r - np.log(np.exp(r).sum(-1, keepdims=True))
    lc = c - np.log(np.exp(c).sum(-1, keepdims=True))
    kl = (pr * (lr - lc)).sum(-1)
    top = (r.argmax(-1) == c.argmax(-1))
    return kl, top


def main():
    ref_dir, cand_dir = sys.argv[1], sys.argv[2]
    kls, tops, n = 0.0, 0, 0
    for rf, cf in zip(sorted(glob.glob(ref_dir + "/*.npz")), sorted(glob.glob(cand_dir + "/*.npz"))):
        r = np.load(rf)["logits"]; c = np.load(cf)["logits"]
        assert r.shape == c.shape, (rf, cf)
        kl, top = kl_from_pair(r, c)
        kls += float(kl.sum()); tops += int(top.sum()); n += kl.size
    print(f"tokens={n} meanKL={kls/n:.6e} top1_agree={100*tops/n:.4f}%")


if __name__ == "__main__":
    main()
