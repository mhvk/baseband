from ..vlbi_base import bcd_encode, bcd_decode
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
