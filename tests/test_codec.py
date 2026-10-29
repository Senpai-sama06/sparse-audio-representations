"""Codec size, reconstruction and invalid-stream checks independent of a study."""
import struct

import numpy as np
import pytest

from lrac.codec import HEADER, encode, decode
from lrac.linear import CosineDictionary


@pytest.mark.parametrize("bits", [4, 6, 8, 12, 16])
@pytest.mark.parametrize("kind,dense", [("dct", False), ("impulse_dct", True)])
def test_bytes_alone_reconstruct_quantized_coefficients_and_count_every_bit(bits, kind, dense):
    bank = CosineDictionary(17, kind)
    z = np.linspace(-.9, 1.0, bank.size) if dense else np.r_[.9, np.zeros(bank.size-2), -.3]
    stream = encode(z, samples=17, sample_rate=16000, dictionary=kind, bits=bits)
    waveform, info = decode(stream)
    # Independent scalar rounding reference; no unquantized z reaches decode.
    step = np.max(np.abs(z))/(2**(bits-1)-1)
    expected = np.rint(z/step)*step
    np.testing.assert_allclose(waveform, bank.synthesis(expected), atol=1e-14)
    assert info["support_mode"] == ("bitmap" if dense else "indices")
    assert info["header_bits"]+info["support_bits"]+info["value_bits"]+info["padding_bits"] == 8*len(stream)
    assert info["bitrate_kbps"] == pytest.approx(8*len(stream)*16000/17/1000)
    assert info["active_coefficients"] == np.count_nonzero(expected)


def test_zero_and_single_coordinate_codes_and_invalid_payloads():
    for n in (1, 17):
        stream = encode(np.zeros(n), samples=n, sample_rate=16000, dictionary="dct", bits=4)
        x, info = decode(stream)
        assert np.count_nonzero(x) == 0 and len(stream) == HEADER.size
        assert info["support_bits"] == info["value_bits"] == 0
    stream = encode(np.array([.25]), samples=1, sample_rate=16000, dictionary="dct", bits=4)
    np.testing.assert_allclose(decode(stream)[0], [.25])
    for bad in (stream[:-1], stream+b"\0", b"BAD!"+stream[4:], b""):
        with pytest.raises(ValueError):
            decode(bad)
    invalid = bytearray(stream)
    struct.pack_into("<d", invalid, HEADER.size-8, float("nan"))
    with pytest.raises(ValueError):
        decode(invalid)
