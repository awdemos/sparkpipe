from __future__ import annotations

import numpy as np


def rne4(values):
    values = np.asarray(values, dtype=np.uint16)
    return (
        (values + np.uint16(3) + ((values >> np.uint16(3)) & np.uint16(1)))
        & np.uint16(0xFFF8)
    ).astype(np.uint16)


def encode(values):
    source = np.ascontiguousarray(values).reshape(-1).astype(np.uint16, copy=False)
    source_exponents = (source >> np.uint16(7)) & np.uint16(0xFF)
    if int(source_exponents.max(initial=0)) == 0xFF:
        raise ValueError("w8lut source contains inf or nan")
    rounded = rne4(source)
    exponents = ((rounded >> np.uint16(7)) & np.uint16(0xFF)).astype(np.uint16)
    if int(exponents.max(initial=0)) == 0xFF:
        raise ValueError("w8lut rounding reached exponent 0xff")
    maximum_exponent = int(exponents.max(initial=0))
    e0 = maximum_exponent - 7 if maximum_exponent >= 7 else 0
    mantissas = ((rounded >> np.uint16(3)) & np.uint16(0xF)).astype(np.uint8)
    signs = ((source >> np.uint16(15)) << np.uint16(7)).astype(np.uint8)
    magnitudes = source & np.uint16(0x7FFF)
    in_window = (exponents > e0) | ((exponents == np.uint16(e0)) & (mantissas > 0))
    threshold = np.uint16(((e0 - 1) << 7) | 0x08) if e0 > 0 else np.uint16(4)
    small_codes = np.where(magnitudes > threshold, signs | np.uint8(1), signs)
    full_codes = (
        signs
        | (((exponents - np.uint16(e0)).astype(np.uint8)) << np.uint8(4))
        | mantissas
    ).astype(np.uint8)
    codes = np.where(in_window, full_codes, small_codes).astype(np.uint8)
    below_count = int(((exponents < e0) & (magnitudes != 0)).sum())
    stats = {
        "element_count": int(source.size),
        "below_window_count": below_count,
        "below_window_ppm": below_count * 1000000 // int(source.size),
        "e0": e0,
    }
    return codes, np.uint16(e0), stats


def decode(codes, e0):
    packed = np.asarray(codes, dtype=np.uint8).astype(np.uint16)
    values = (
        ((packed & np.uint16(0x80)) << np.uint16(8))
        | ((np.uint16(e0) + ((packed >> np.uint16(4)) & np.uint16(7))) << np.uint16(7))
        | ((packed & np.uint16(0xF)) << np.uint16(3))
    )
    signed_zero = (packed & np.uint16(0x80)) << np.uint16(8)
    return np.where((packed & np.uint16(0x7F)) == 0, signed_zero, values).astype(np.uint16)


def verify(values, codes, e0):
    source = np.ascontiguousarray(values).reshape(-1).astype(np.uint16, copy=False)
    rounded = rne4(source)
    exponents = ((rounded >> np.uint16(7)) & np.uint16(0xFF)).astype(np.uint16)
    mantissas = ((rounded >> np.uint16(3)) & np.uint16(0xF)).astype(np.uint16)
    in_window = (exponents > int(e0)) | ((exponents == np.uint16(e0)) & (mantissas != 0))
    decoded = decode(codes, e0)
    if not np.array_equal(decoded[in_window], rounded[in_window]):
        raise ValueError("w8lut in-window decode mismatch")
    reencoded, re_e0, _ = encode(decoded)
    if int(re_e0) != int(e0) or not np.array_equal(reencoded, codes):
        raise ValueError("w8lut decode and re-encode is not idempotent")


def bf16_to_f32(values):
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
