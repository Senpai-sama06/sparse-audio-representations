"""A small, versioned sparse bitstream with explicit support and value costs.

This deliberately simple codec measures actual bytes. It uses no entropy coder,
trained probability model, hidden normalization or sample-specific dictionary.
All integers in the header are little-endian; payload bit order is MSB first.
"""
import struct

import numpy as np

from lrac.linear import CosineDictionary


# magic, version, dictionary ID, coefficient bits, support mode, N, fs, K, step.
# The binary64 quantizer step is transmitted and used by BOTH encoder and decoder.
HEADER = struct.Struct("<4sBBBBIIId")
DICTIONARIES = {"dct": 0, "impulse_dct": 1}


def _pack(values, width):
    """Pack unsigned integers without charging a byte per coefficient or index."""
    values = np.asarray(values, dtype=np.uint64)
    if width == 0:
        return b""
    shifts = np.arange(width-1, -1, -1, dtype=np.uint64)
    bits = ((values[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits, bitorder="big").tobytes()


def _unpack(payload, count, width):
    if width == 0:
        return np.zeros(count, dtype=np.int64)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    # Padding is part of measured rate and must be zero in this canonical format.
    if bits[count*width:].any():
        raise ValueError("Nonzero bitstream padding")
    matrix = bits[:count*width].reshape(count, width).astype(np.uint64)
    weights = 1 << np.arange(width-1, -1, -1, dtype=np.uint64)
    return (matrix @ weights).astype(np.int64)


def encode(coefficients, *, samples, sample_rate, dictionary, bits):
    """Quantize and serialize coefficients; return the complete byte string.

    Quantization is symmetric round-to-nearest, with signed levels -qmax..qmax.
    Quantized zeros are omitted. The unused most-negative code is rejected by
    the decoder rather than silently assigning it a different interpretation.
    """
    if bits not in {4, 6, 8, 12, 16} or not 0 < sample_rate < 2**32:
        raise ValueError("Unsupported quantizer width or sample rate")
    bank = CosineDictionary(samples, dictionary)
    z = np.asarray(coefficients, dtype=np.float64)
    if z.shape != (bank.size,) or not np.isfinite(z).all() or samples >= 2**32:
        raise ValueError("Invalid coefficient vector or sample count")
    qmax = 2**(bits-1)-1
    peak = float(np.abs(z).max())
    step = peak/qmax if peak else 1.0
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Coefficient scale cannot be represented")
    quantized = np.clip(np.rint(z/step), -qmax, qmax).astype(np.int64)
    positions = np.flatnonzero(quantized)
    index_width = (bank.size-1).bit_length()
    # Compare actual padded support payloads, with ties resolved toward indices.
    bitmap_bytes = (bank.size+7)//8
    list_bytes = (len(positions)*index_width+7)//8
    mode = int(bitmap_bytes < list_bytes)
    support = _pack(quantized != 0, 1) if mode else _pack(positions, index_width)
    values = _pack(quantized[positions] + qmax, bits)
    header = HEADER.pack(b"SAC1", 1, DICTIONARIES[dictionary], bits, mode,
                         samples, sample_rate, len(positions), step)
    return header + support + values


def decode(payload):
    """Decode using bytes alone; return waveform and a complete bit accounting.

    No source waveform, coefficient scale, lambda or support array is accepted.
    Lengths are validated before allocating coefficient arrays.
    """
    if len(payload) < HEADER.size:
        raise ValueError("Truncated sparse header")
    magic, version, bank_id, bits, mode, n, sample_rate, count, step = HEADER.unpack_from(payload)
    if magic != b"SAC1" or version != 1 or bank_id not in {0, 1}:
        raise ValueError("Unknown sparse format or dictionary")
    if bits not in {4, 6, 8, 12, 16} or mode not in {0, 1} or not n or not sample_rate:
        raise ValueError("Invalid sparse header")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Invalid transmitted quantizer step")
    dictionary = "dct" if bank_id == 0 else "impulse_dct"
    bank = CosineDictionary(n, dictionary)
    if count > bank.size:
        raise ValueError("Support exceeds dictionary size")
    index_width = (bank.size-1).bit_length()
    support_bits = bank.size if mode else count*index_width
    support_bytes, value_bytes = (support_bits+7)//8, (count*bits+7)//8
    if len(payload) != HEADER.size + support_bytes + value_bytes:
        raise ValueError("Truncated or trailing sparse payload")
    split = HEADER.size + support_bytes
    if mode:
        mask = _unpack(payload[HEADER.size:split], bank.size, 1)
        positions = np.flatnonzero(mask)
    else:
        positions = _unpack(payload[HEADER.size:split], count, index_width)
    if len(positions) != count or (count and (positions[-1] >= bank.size or np.any(np.diff(positions) <= 0))):
        raise ValueError("Invalid or duplicate support positions")
    qmax = 2**(bits-1)-1
    quantized = _unpack(payload[split:], count, bits) - qmax
    if np.any(quantized == 0) or np.any(np.abs(quantized) > qmax):
        raise ValueError("Invalid nonzero coefficient code")
    z = np.zeros(bank.size)
    z[positions] = quantized * step
    total_bits = 8*len(payload)
    return bank.synthesis(z), dict(
        dictionary=dictionary, samples=n, sample_rate=sample_rate, bits=bits,
        active_coefficients=count, coefficient_slots=bank.size,
        support_mode="bitmap" if mode else "indices", quantizer_step=step,
        header_bits=8*HEADER.size, support_bits=support_bits, value_bits=count*bits,
        padding_bits=total_bits-8*HEADER.size-support_bits-count*bits,
        total_bits=total_bits, bitrate_kbps=total_bits*sample_rate/n/1000,
        bits_per_sample=total_bits/n,
    )
