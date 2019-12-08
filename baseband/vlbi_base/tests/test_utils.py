# Licensed under the GPLv3 - see LICENSE
import pytest
import numpy as np

from ..utils import bcd_encode, bcd_decode, CRC, lcm


class TestBCD:
    def test_bcd_decode(self):
        assert bcd_decode(0x1) == 1
        assert bcd_decode(0x9123) == 9123
        with pytest.raises(ValueError):
            bcd_decode(0xf)
        decoded = bcd_decode(np.array([0x1, 0x9123]))
        assert isinstance(decoded, np.ndarray)
        assert np.all(decoded == np.array([1, 9123]))
        with pytest.raises(ValueError):
            bcd_decode(np.array([0xf, 9123]))
        with pytest.raises(TypeError):
            bcd_decode([1, 2])

    def test_bcd_encode(self):
        assert bcd_encode(1) == 0x1
        assert bcd_encode(9123) == 0x9123
        with pytest.raises(TypeError):
            bcd_encode('bla')

    def test_roundtrip(self):
        assert bcd_decode(bcd_encode(15)) == 15
        assert bcd_decode(bcd_encode(8765)) == 8765
        a = np.array([1, 9123])
        assert np.all(bcd_decode(bcd_encode(a)) == a)


def test_crc():
    # Test example from age 4 of
    # http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
    stream = '0000 002D 0330 0000' + 'FFFF FFFF' + '4053 2143 3805 5'
    crc_expected = '284'
    crc12 = CRC(0x180f)
    stream = stream.replace(' ', '').lower()
    istream = int(stream, base=16)
    assert '{:037x}'.format(istream) == stream
    bitstream = np.array([((istream & (1 << bit)) != 0)
                          for bit in range(37*4-1, -1, -1)], np.bool)
    crcstream = crc12(bitstream)
    crc = np.bitwise_or.reduce(crcstream.astype(np.uint32)
                               << np.arange(11, -1, -1))
    assert '{:03x}'.format(crc) == crc_expected
    fullstream = np.hstack((bitstream, crcstream))
    assert crc12.check(fullstream)


@pytest.mark.parametrize(
    ('a', 'b', 'lcm_out'),
    ((7, 14, 14),
     (7853, 6199, 48680747),
     (0, 5, 0),
     (4, -12, 12),
     (-4, -12, 12)))
def test_lcm(a, b, lcm_out):
    assert lcm(a, b) == lcm_out
