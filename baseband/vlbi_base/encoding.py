# Licensed under the GPLv3 - see LICENSE
"""Encoders and decoders for generic VLBI data formats."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np


__all__ = ['OPTIMAL_2BIT_HIGH', 'TWO_BIT_1_SIGMA', 'FOUR_BIT_1_SIGMA',
           'EIGHT_BIT_1_SIGMA', 'decoder_levels', 'encode_2bit_base',
           'encode_4bit_base', 'decode_8bit', 'encode_8bit']


# The high mag value for 2-bit reconstruction.  Note that mark5access uses
# OPTIMAL_2BIT_HIGH = 3.3359, which is possibly a typo.
OPTIMAL_2BIT_HIGH = 3.316505
r"""Optimal high value for a 2-bit digitizer for which the low value is 1.

It is chosen such that for a normal distribution in which 68.269% of all
values are at the low level, this is the mean of the others, i.e.,

.. math::

    l = \frac{\int_\sigma^\infty x \exp(-\frac{x^2}{2\sigma^2}) dx}
             {\int_\sigma^\infty \exp(-\frac{x^2}{2\sigma^2}) dx},

where the standard deviation is determined from:

.. math::

    1 = \frac{\int_0^\sigma x \exp(-\frac{x^2}{2\sigma^2}) dx}
         {\int_0^\sigma \exp(-\frac{x^2}{2\sigma^2}) dx}.

These give:

.. math::

    \sigma = \frac{\sqrt{\frac{\pi}{2}}\mathrm{erf}
             (\sqrt{1/2})}{1 - \sqrt{1/e}} = 2.174564,

and

.. math::

    l = \frac{1}{(\sqrt{e} - 1)(1/\mathrm{erf}(\sqrt{1/2}) - 1)} = 3.316505
"""
TWO_BIT_1_SIGMA = 2.174564
"""Optimal level between low and high for the above OPTIMAL_2BIT_HIGH."""
FOUR_BIT_1_SIGMA = 2.95
"""Scaling for four-bit encoding that makes it look like 2 bit."""
EIGHT_BIT_1_SIGMA = 71.0 / 2.
"""Scaling for eight-bit encoding that makes it look like 2 bit."""

decoder_levels = {
    1: np.array([-1.0, 1.0], dtype=np.float32),
    2: np.array([-OPTIMAL_2BIT_HIGH, -1.0, 1.0, OPTIMAL_2BIT_HIGH],
                dtype=np.float32),
    4: (np.arange(16, dtype=np.float32) - 8.) / FOUR_BIT_1_SIGMA}
"""Levels for data encoded with different numbers of bits.."""

two_bit_2_sigma = 2 * TWO_BIT_1_SIGMA
clip_low, clip_high = -1.5 * TWO_BIT_1_SIGMA, 1.5 * TWO_BIT_1_SIGMA


def encode_2bit_base(values):
    """Generic encoder for data stored using two bits.

    This returns an unsigned integer array containing encoded sample
    values that range from 0 to 3.  The conversion from floating point
    sample value to unsigned int is given below, with
    ``lv = TWO_BIT_1_SIGMA = 2.1745``:

      ================= ======
      Input range       Output
      ================= ======
            value < -lv   0
      -lv < value <  0.   2
       0. < value <  lv   1
       lv < value         3
      ================= ======

    This does not pack the samples into bytes.
    """
    # Optimized for speed by doing calculations in-place, and ensuring that
    # the dtypes match.
    values = np.clip(values, clip_low, clip_high)
    values += two_bit_2_sigma
    bitvalues = np.empty(values.shape, np.uint8)
    return np.floor_divide(values, TWO_BIT_1_SIGMA, out=bitvalues,
                           casting='unsafe')


def encode_4bit_base(values):
    """Generic encoder for data stored using four bits.

    This returns an unsigned integer array containing encoded sample
    values that range from 0 to 15.  Floating point sample values are
    converted to unsigned int by first scaling them by
    ``FOUR_BIT_1_SIGMA = 2.95``, then adding 8.5 (the 0.5 to ensure proper
    rounding when typecasting to uint8). Some sample output levels are:

      ========================= ======
      Input range               Output
      ========================= ======
             value*scale < -7.5    0
      -7.5 < value*scale < -6.5    1
      -0.5 < value*scale < +0.5    8
       6.5 < value*scale          15
      ========================= ======

    This does not pack the samples into bytes.
    """
    # Optimized for speed by doing calculations in-place.
    values = values * FOUR_BIT_1_SIGMA
    values += 8.5
    return np.clip(values, 0., 15., out=values).astype(np.uint8)


def decode_8bit(words):
    """Generic decoder for data stored using 8 bits.

    We follow mark5access, which assumes the values 0 to 255 encode
    -127.5 to 127.5, scaled down to match 2 bit data by a factor of 35.5
    (`~baseband.vlbi_base.encoding.EIGHT_BIT_1_SIGMA`)

    For comparison, GMRT phased data treats the 8-bit data values simply
    as signed integers.
    """
    b = words.view(np.uint8).astype(np.float32)
    b -= 127.5
    b /= EIGHT_BIT_1_SIGMA
    return b


def encode_8bit(values):
    """Encode 8 bit VDIF data.

    We follow mark5access, which assumes the values 0 to 255 encode
    -127.5 to 127.5, scaled down to match 2 bit data by a factor of 35.5
    (`~baseband.vlbi_base.encoding.EIGHT_BIT_1_SIGMA`)

    For comparison, GMRT phased data treats the 8-bit data values simply
    as signed integers.
    """
    return (np.clip(np.rint(values * EIGHT_BIT_1_SIGMA + 127.5), 0, 255)
            .astype(np.uint8))
