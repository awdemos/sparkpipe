"""W8LUT v2 numpy encoder. Bit-identical to w8lut_ref.c (cross-checked)."""
import numpy as np


def rne4(b):
    b = b.astype(np.uint16)
    return ((b + np.uint16(3) + ((b >> np.uint16(3)) & np.uint16(1))) & np.uint16(0xFFF8)).astype(np.uint16)


def encode(w_u16):
    w = np.ascontiguousarray(w_u16).reshape(-1).astype(np.uint16)
    if int(((w >> 7) & np.uint16(0xFF)).max(initial=0)) == 0xFF:
        raise ValueError("w8lut: inf/nan in source (-11)")
    r = rne4(w)
    e = ((r >> np.uint16(7)) & np.uint16(0xFF)).astype(np.uint16)
    if int(e.max(initial=0)) == 0xFF:
        raise ValueError("w8lut: rounding reached exp 0xFF (-12)")
    e7 = int(e.max(initial=0))
    e0 = e7 - 7 if e7 >= 7 else 0
    m = ((r >> np.uint16(3)) & np.uint16(0xF)).astype(np.uint8)
    s = ((w >> np.uint16(15)) << np.uint16(7)).astype(np.uint8)
    mag = (w & np.uint16(0x7FFF))
    inwin = (e > e0) | ((e == np.uint16(e0)) & (m > 0))
    T = np.uint16(((e0 - 1) << 7) | 0x08) if e0 > 0 else np.uint16(4)
    small = np.where(mag > T, s | np.uint8(1), s)
    full = (s | (((e - np.uint16(e0)).astype(np.uint8)) << np.uint8(4)) | m).astype(np.uint8)
    codes = np.where(inwin, full, small).astype(np.uint8)
    below = int(((e < e0) & (mag != 0)).sum())
    return codes, np.uint16(e0), {"n": int(w.size), "below_ppm": below * 1000000 // int(w.size), "e0": e0}


def decode(codes, e0):
    c = codes.astype(np.uint16)
    val = ((c & np.uint16(0x80)) << np.uint16(8)) | ((np.uint16(e0) + ((c >> np.uint16(4)) & np.uint16(7))) << np.uint16(7)) | ((c & np.uint16(0xF)) << np.uint16(3))
    return np.where((c & np.uint16(0x7F)) == 0, ((c & np.uint16(0x80)) << np.uint16(8)).astype(np.uint16), val.astype(np.uint16))


def bf16_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def e4m3_block128_dequant(w_u16, rows, cols):
    from ml_dtypes import float8_e4m3fn
    f = bf16_to_f32(w_u16).reshape(rows, cols).astype(np.float32)
    out = np.empty_like(f)
    for r0 in range(0, rows, 128):
        for c0 in range(0, cols, 128):
            blk = f[r0:r0 + 128, c0:c0 + 128]
            amax = float(np.abs(blk).max(initial=0.0))
            sc = amax / 448.0 if amax > 0 else 1.0
            out[r0:r0 + 128, c0:c0 + 128] = (blk / sc).astype(float8_e4m3fn).astype(np.float32) * sc
    return out
