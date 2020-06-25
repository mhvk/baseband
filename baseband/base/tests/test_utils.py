# Licensed under the GPLv3 - see LICENSE
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from ..utils import lcm, bcd_encode, bcd_decode, byte_array, CRC, CRCStack


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


class TestCRC12:

    @staticmethod
    def hex_to_stream(string):
        n = len(string) * 4
        scalar = int(string, base=16)
        return np.array([((scalar & (1 << bit)) != 0)
                         for bit in range(n-1, -1, -1)], np.bool)

    @staticmethod
    def stream_to_scalar(stream):
        n = len(stream)
        result = 0
        for i, s in enumerate(stream):
            result += int(s) << (n - i - 1)

        return result

    def setup(self):
        # Test example from page 4 of
        # https://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
        stream_hex = '0000 002D 0330 0000' + 'FFFF FFFF' + '4053 2143 3805 5'
        self.crc_hex = '284'
        self.crc12 = CRC(0x180f)
        self.crcstack12 = CRCStack(0x180f)

        self.stream_hex = stream_hex.replace(' ', '').lower()

        self.stream = int(self.stream_hex, base=16)
        self.bitstream = self.hex_to_stream(self.stream_hex)

        self.crc = int(self.crc_hex, base=16)
        self.crcstream = self.hex_to_stream(self.crc_hex)

    def test_setup(self):
        assert '{:037x}'.format(self.stream) == self.stream_hex
        assert '{:03x}'.format(self.crc) == self.crc_hex
        assert self.stream_to_scalar(self.bitstream) == self.stream
        assert self.stream_to_scalar(self.crcstream) == self.crc

    def test_crc_stream(self):
        crcstream = self.crcstack12(self.bitstream)
        assert np.all(crcstream == self.crcstream)

    def test_check_crc_stream(self):
        fullstream = np.hstack((self.bitstream, self.crcstream))
        assert self.crcstack12.check(fullstream)

    def test_crc_scalar(self):
        crc = self.crc12(self.stream)
        assert crc == self.crc

    def test_check_crc_scalar(self):
        scalar = (self.stream << len(self.crc12)) + self.crc
        assert self.crc12.check(scalar)

    def test_crc_array(self):
        scalar = 0x12345678
        expected = self.crc12(scalar)
        array = (scalar * np.ones(10, dtype=int)).astype('u8')
        crc = self.crc12(array)
        assert crc.shape == array.shape
        assert np.all(crc == expected)

    def test_check_crc_array(self):
        scalar = 0x12345678
        scalar = (scalar << len(self.crc12)) + self.crc12(scalar)
        array = (scalar * np.ones(10, dtype=int)).astype('u8')
        check = self.crc12.check(array)
        assert check.shape == array.shape
        assert np.all(check)


@pytest.mark.parametrize(
    ('a', 'b', 'lcm_out'),
    ((7, 14, 14),
     (7853, 6199, 48680747),
     (0, 5, 0),
     (4, -12, 12),
     (-4, -12, 12)))
def test_lcm(a, b, lcm_out):
    assert lcm(a, b) == lcm_out


@pytest.mark.parametrize(
    ('pattern', 'expected'),
    [(b'\xa0\x55', [160, 85]),
     (0x55a0, [160, 85, 0, 0]),
     ([0x55, 0xa0], [85, 0, 0, 0, 160, 0, 0, 0]),
     (np.array([0xa0, 0x55], 'u1'), [160, 85]),
     (np.array(0x55a0, '<u4'), [160, 85, 0, 0]),
     (np.array(0x55a0, '<u8'), [160, 85, 0, 0, 0, 0, 0, 0]),
     (np.array([0x55, 0xa0], '<u4'), [85, 0, 0, 0, 160, 0, 0, 0])])
def test_byte_array(pattern, expected):
    result = byte_array(pattern)
    expected = np.array(expected, 'u1')
    assert_array_equal(result, expected)
