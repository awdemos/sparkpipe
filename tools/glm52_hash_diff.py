#!/usr/bin/env python3
import re
import sys

LINE = re.compile(r"hidden_tcp_(send_header|deliver) seq=(\d+) token=(\d+) .*?hidden_hash=([0-9a-f]+) sideband_hash=([0-9a-f]+) hidden_bytes=(\d+) sideband_bytes=(\d+)")

def fnv64(data, seed=0):
    h = seed ^ 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h

ZERO_HASH_CACHE = {}

def zero_hash(size):
    if size not in ZERO_HASH_CACHE:
        ZERO_HASH_CACHE[size] = format(fnv64(bytes(size)), "016x")
    return ZERO_HASH_CACHE[size]

def load_run(pairs):
    run = {}
    for pair in pairs:
        rank_text, path = pair.split(":", 1)
        rank = int(rank_text)
        for line in open(path, errors="replace"):
            m = LINE.search(line)
            if m is None:
                continue
            kind = "tx" if m.group(1) == "send_header" else "rx"
            key = (int(m.group(3)), int(m.group(2)), rank, kind)
            run[key] = (m.group(4), m.group(5), int(m.group(6)), int(m.group(7)))
    return run

def chain_report(run):
    ranks = sorted(set(k[2] for k in run.keys()))
    tokens = sorted(set((k[0], k[1]) for k in run.keys()))
    findings = 0
    for token, seq in tokens:
        for rank in ranks:
            rx = run.get((token, seq, rank, "rx"))
            tx = run.get((token, seq, rank, "tx"))
            flags = []
            if rx is not None and rx[0] == zero_hash(rx[2]):
                flags.append("RX_ZEROS")
            if tx is not None and tx[0] == zero_hash(tx[2]):
                flags.append("TX_ZEROS")
            if rx is not None and tx is not None and rx[0] == tx[0]:
                flags.append("PASSTHROUGH")
            if flags:
                findings += 1
            rx_text = rx[0] if rx is not None else "-"
            tx_text = tx[0] if tx is not None else "-"
            print(f"chain	token={token}	seq={seq}	rank={rank}	rx={rx_text}	tx={tx_text}	{'+'.join(flags) if flags else 'transform'}")
    print(f"chain	findings={findings}")
    return findings

def hop_integrity(name, run):
    bad = 0
    for (token, seq, rank, kind), value in sorted(run.items()):
        if kind != "tx":
            continue
        rx = run.get((token, seq, rank + 1, "rx"))
        if rx is None:
            print(f"{name}\thop_missing_rx\ttoken={token}\tseq={seq}\t{rank}->{rank+1}\ttx={value[0]}/{value[1]}")
            bad += 1
        elif rx != value:
            print(f"{name}\thop_mismatch\ttoken={token}\tseq={seq}\t{rank}->{rank+1}\ttx={value[0]}/{value[1]}\trx={rx[0]}/{rx[1]}")
            bad += 1
    if bad == 0:
        print(f"{name}\thop_integrity_ok")
    return bad

def cross_diff(a, b):
    keys = sorted(k for k in a.keys() if k[3] == "tx")
    for key in keys:
        token, seq, rank, _ = key
        if key not in b:
            print(f"cross\tmissing_in_b\ttoken={token}\tseq={seq}\trank={rank}")
            return 1
        if a[key] != b[key]:
            print(f"cross\tFIRST_DIVERGENCE\ttoken={token}\tseq={seq}\trank={rank}\ta={a[key][0]}/{a[key][1]}\tb={b[key][0]}/{b[key][1]}")
            return 1
    print("cross\tidentical")
    return 0

def selftest():
    import tempfile, os
    tx = "hidden_tcp_send_header seq=1 token=0 active=1 sideband_kind=0 sideband_bps=0 hidden_hash=00000000000000aa sideband_hash=0000000000000000 hidden_bytes=12288 sideband_bytes=0 total=12800\n"
    rx = "hidden_tcp_deliver seq=1 token=0 active=1 sideband_kind=0 hidden_hash=00000000000000aa sideband_hash=0000000000000000 hidden_bytes=12288 sideband_bytes=0\n"
    rx_bad = rx.replace("aa", "ab")
    tx_bad = tx.replace("aa", "ab")
    d = tempfile.mkdtemp()
    open(f"{d}/r0", "w").write(tx)
    open(f"{d}/r1", "w").write(rx)
    open(f"{d}/r1b", "w").write(rx_bad)
    open(f"{d}/r0b", "w").write(tx_bad)
    good = load_run([f"0:{d}/r0", f"1:{d}/r1"])
    hop_bad = load_run([f"0:{d}/r0", f"1:{d}/r1b"])
    tx_diverged = load_run([f"0:{d}/r0b", f"1:{d}/r1b"])
    assert hop_integrity("good", good) == 0
    assert hop_integrity("bad", hop_bad) == 1
    assert cross_diff(good, good) == 0
    assert cross_diff(good, tx_diverged) == 1
    zh = zero_hash(4)
    zline = f"hidden_tcp_send_header seq=1 token=1 active=1 sideband_kind=0 sideband_bps=0 hidden_hash={zh} sideband_hash={zero_hash(0) if False else '0'*16} hidden_bytes=4 sideband_bytes=0 total=64\n"
    pline_rx = "hidden_tcp_deliver seq=1 token=2 active=1 sideband_kind=0 hidden_hash=00000000000000cc sideband_hash=0000000000000000 hidden_bytes=4 sideband_bytes=0\n"
    pline_tx = "hidden_tcp_send_header seq=1 token=2 active=1 sideband_kind=0 sideband_bps=0 hidden_hash=00000000000000cc sideband_hash=0000000000000000 hidden_bytes=4 sideband_bytes=0 total=64\n"
    open(f"{d}/rz", "w").write(zline + pline_rx + pline_tx)
    zrun = load_run([f"5:{d}/rz"])
    assert chain_report(zrun) == 2
    print("selftest_ok")

def main():
    argv = sys.argv[1:]
    if argv == ["--selftest"]:
        selftest()
        return 0
    if "--b" in argv:
        split = argv.index("--b")
        if argv[0] != "--a":
            print("usage: glm52_hash_diff.py --a RANK:PATH... --b RANK:PATH... | --a RANK:PATH... | --selftest")
            return 2
        a = load_run(argv[1:split])
        b = load_run(argv[split + 1:])
        bad = hop_integrity("a", a)
        bad += hop_integrity("b", b)
        bad += cross_diff(a, b)
        return 1 if bad != 0 else 0
    if argv and argv[0] == "--chain":
        a = load_run(argv[1:])
        chain_report(a)
        return 1 if hop_integrity("chain", a) != 0 else 0
    if argv and argv[0] == "--a":
        a = load_run(argv[1:])
        return 1 if hop_integrity("a", a) != 0 else 0
    print("usage: glm52_hash_diff.py --a RANK:PATH... --b RANK:PATH... | --chain RANK:PATH... | --a RANK:PATH... | --selftest")
    return 2

if __name__ == "__main__":
    sys.exit(main())
