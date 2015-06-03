from ..vlbi_base import bcd_encode, bcd_decode, CRC
import numpy as np
from astropy.tests.helper import pytest


class TestBCD(object):
    def test_bcd_decode(self):
        assert bcd_decode(0x1) == 1
        assert bcd_decode(0x9123) == 9123
        with pytest.raises(ValueError):
            bcd_decode(0xf)

    def test_bcd_encode(self):
        assert bcd_encode(1) == 0x1
        assert bcd_encode(9123) == 0x9123

    def test_roundtrip(self):
        assert bcd_decode(bcd_encode(15)) == 15
        assert bcd_decode(bcd_encode(8765)) == 8765


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
    crc = np.bitwise_or.reduce(crcstream.astype(np.uint32) <<
                               np.arange(11, -1, -1))
    assert '{:03x}'.format(crc) == crc_expected
    fullstream = np.hstack((bitstream, crcstream))
    assert crc12.check(fullstream)
