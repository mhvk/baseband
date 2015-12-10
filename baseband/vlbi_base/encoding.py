"""Encoders and decoders for generic VLBI data formats."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np


__all__ = ['OPTIMAL_2BIT_HIGH', 'TWO_BIT_1_SIGMA', 'FOUR_BIT_1_SIGMA',
           'decoder_levels', 'encode_2bit_real_base']


# The high mag value for 2-bit reconstruction.
OPTIMAL_2BIT_HIGH = 3.3359
r"""Optimal high value for a 2-bit digitizer for which the low value is 1.

It is chosen such that for a normal distribution in which 68.269% of all values
are at the low level, this is the mean of the others, i.e.,

.. math::

     l = \frac{\int_1^\infty x \exp(-\frac12x^2) dx}
              {\int_0^1 x \exp(-\frac12x^2) dx} \times
         \frac{\int_0^1 \exp(-\frac12x^2)dx}
              {\int_1^\infty \exp(-\frac12x^2) dx}

Note that for this value, the standard deviation is 2.1745.
"""
TWO_BIT_1_SIGMA = 2.1745
"""Optimal level between low and high for the above OPTIMAL_2BIT_HIGH."""
FOUR_BIT_1_SIGMA = 2.95
"""Level for four-bit encoding."""


decoder_levels = {
    1: np.array([-1.0, 1.0], dtype=np.float32),
    2: np.array([-OPTIMAL_2BIT_HIGH, -1.0, 1.0, OPTIMAL_2BIT_HIGH],
                dtype=np.float32),
    4: (np.arange(16) - 8.)/FOUR_BIT_1_SIGMA}
"""Levels for data encoded with different numbers of bits.."""

two_bit_2_sigma = 2 * TWO_BIT_1_SIGMA
clip_low, clip_high = -1.5 * TWO_BIT_1_SIGMA, 1.5 * TWO_BIT_1_SIGMA


def encode_2bit_real_base(values):
    """Generic encoder for data stored using two bits.

    This returns an unsigned integer array with values ranging from 0 to 3.
    It does not do the merging of samples together.

    Effectively, get indices such that for ``lv=TWO_BIT_1_SIGMA=2.1745``:
      ================= ======
      Input range       Output
      ================= ======
            value < -lv   0
      -lv < value <  0.   2
       0. < value <  lv   1
       lv < value         3
      ================= ======
    """
    # Optimized for speed by doing calculations in-place, and ensuring that
    # the dtypes match.
    values = np.clip(values, clip_low, clip_high)
    values += two_bit_2_sigma
    bitvalues = np.empty(values.shape, np.uint8)
    return np.floor_divide(values, TWO_BIT_1_SIGMA, out=bitvalues,
                           casting='unsafe')


def decode_8bit_real(words, out=None):
    """Generic decoder for data stored using 8 bits.

    We assume bytes encode -128 to 127, i.e., their direct values.
    This is the same as assumed in MWA beam-combined data and in GMRT
    phased data, but in contrast to mark5access, which assumes they
    represent -127.5 .. 127.5, i.e., symmetric around zero.
    """
    b = words.view(np.int8)
    if out is None:
        return b.astype(np.float32)
    else:
        outf4 = out.reshape(-1)
        assert outf4.base is out or outf4.base is out.base
        outf4[:] = b
        return out


def decode_8bit_complex(words, out=None):
    """Generic decoder for complex data stored using 8 bits per component."""
    if out is None:
        return decode_8bit_real(words, out).view(np.complex64)
    else:
        decode_8bit_real(words, out.view(out.real.dtype))
        return out


def encode_8bit_real(values):
    """Encode 8 bit VDIF data.

    We assume bytes encode -128 to 127, i.e., their direct values.
    This is the same as assumed in MWA beam-combined data, but in contrast
    to mark5access, which assumes they represent -127.5 .. 127.5, i.e.,
    symmetric around zero.
    """
    return np.clip(np.round(values), -128, 127).astype(np.int8)


def encode_8bit_complex(values):
    return encode_8bit_real(values.view(values.real.dtype))
