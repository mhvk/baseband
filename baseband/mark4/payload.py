# Licensed under the GPLv3 - see LICENSE
"""
Definitions for VLBI Mark 4 payloads.

Implements a Mark4Payload class used to store payload words, and decode to
or encode from a data array.

For the specification, see
https://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
"""
import sys
from collections import namedtuple

import numpy as np

from ..base.payload import PayloadBase
from ..base.encoding import encode_2bit_base, decoder_levels
from ..base.utils import fixedvalue
from .header import MARK4_DTYPES


__all__ = ['reorder32', 'reorder64', 'init_luts', 'decode_8chan_2bit_fanout4',
           'encode_8chan_2bit_fanout4', 'Mark4Payload']

#  2bit/fanout4 use the following in decoding 32 and 64 track data:
if sys.byteorder == 'big':  # pragma: no cover
    def reorder32(x):
        """Reorder 32-track bits to bring signs & magnitudes together."""
        return (((x & 0x55AA55AA))
                | ((x & 0xAA00AA00) >> 9)
                | ((x & 0x00550055) << 9))

    def reorder64(x):
        """Reorder 64-track bits to bring signs & magnitudes together."""
        return (((x & 0x55AA55AA55AA55AA))
                | ((x & 0xAA00AA00AA00AA00) >> 9)
                | ((x & 0x0055005500550055) << 9))

    def reorder64_Ft(x):
        """Reorder 64-track bits to bring signs & magnitudes together.

        Special version for the Ft station, which has unusual settings.
        """
        return (((x & 0xAFFAFFFFAFFAFFFF))
                | ((x & 0x0005000000050000) << 12)
                | ((x & 0x5000000050000000) >> 12))
else:
    def reorder32(x):
        """Reorder 32-track bits to bring signs & magnitudes together."""
        return (((x & 0xAA55AA55))
                | ((x & 0x55005500) >> 7)
                | ((x & 0x00AA00AA) << 7))

    # Can speed this up from 140 to 132 us by predefining bit patterns as
    # array scalars.  Inplace calculations do not seem to help much.
    def reorder64(x):
        """Reorder 64-track bits to bring signs & magnitudes together."""
        return (((x & 0xAA55AA55AA55AA55))
                | ((x & 0x5500550055005500) >> 7)
                | ((x & 0x00AA00AA00AA00AA) << 7))

    def reorder64_Ft(x):
        """Reorder 64-track bits to bring signs & magnitudes together.

        Special version for the Ft station, which has unusual settings.
        """
        return (((x & 0xFFFFFAAFFFFFFAAF))
                | ((x & 0x0000050000000500) >> 4)
                | ((x & 0x0000005000000050) << 4))
    # Check on 2015-JUL-12: C code: 738811025863578102 -> 738829572664316278
    # 118, 209, 53, 244, 148, 217, 64, 10
    # reorder64(np.array([738811025863578102], dtype=np.uint64))
    # # array([738829572664316278], dtype=uint64)
    # reorder64(np.array([738811025863578102], dtype=np.uint64)).view(np.uint8)
    # # array([118, 209,  53, 244, 148, 217,  64,  10], dtype=uint8)
    # decode_2bit_64track_fanout4(
    #     np.array([738811025863578102], dtype=np.int64)).astype(int).T
    # -1  1  3  1  array([[-1,  1,  3,  1],
    #  1  1  3 -3         [ 1,  1,  3, -3],
    #  1 -3  1  3         [ 1, -3,  1,  3],
    # -3  1  3  3         [-3,  1,  3,  3],
    # -3  1  1 -1         [-3,  1,  1, -1],
    # -3 -3 -3  1         [-3, -3, -3,  1],
    #  1 -1  1  3         [ 1, -1,  1,  3],
    # -1 -1 -3 -3         [-1, -1, -3, -3]])


def init_luts():
    """Set up the look-up tables for levels as a function of input byte."""
    # Organisation by bits is quite odd for Mark 4.
    b = np.arange(256)[:, np.newaxis]
    # lut1bit
    i = np.arange(8)
    # For all 1-bit modes; if set, sign=-1, so need to get item 0.
    lut1bit = decoder_levels[1][((b >> i) & 1) ^ 1]
    i = np.arange(4)
    # fanout 1 @ 8/16t, fanout 4 @ 32/64t
    s = i*2  # 0, 2, 4, 6
    m = s+1  # 1, 3, 5, 7
    lut2bit1 = decoder_levels[2][2*(b >> s & 1)
                                 + (b >> m & 1)]
    # fanout 2 @ 8/16t, fanout 1 @ 32/64t
    s = i + (i//2)*2  # 0, 1, 4, 5
    m = s + 2         # 2, 3, 6, 7
    lut2bit2 = decoder_levels[2][2*(b >> s & 1)
                                 + (b >> m & 1)]
    # fanout 4 @ 8/16t, fanout 2 @ 32/64t
    s = i    # 0, 1, 2, 3
    m = s+4  # 4, 5, 6, 7
    lut2bit3 = decoder_levels[2][2*(b >> s & 1)
                                 + (b >> m & 1)]
    return lut1bit, lut2bit1, lut2bit2, lut2bit3


lut1bit, lut2bit1, lut2bit2, lut2bit3 = init_luts()

# Look-up table for the number of bits in a byte.
nbits = ((np.arange(256)[:, np.newaxis] >> np.arange(8) & 1)
         .sum(1).astype(np.int16))


def decode_2chan_2bit_fanout4(frame):
    """Decode payload for 2 channels using 2 bits, fan-out 4 (16 tracks)."""
    # header['magnitude_bit'] = 00001111,00001111
    # makes sense with lut2bit3
    # header['fan_out'] = 01230123,01230123
    # header['converter_id'] = 00000000,11111111
    # header['lsb_output'] = 11111111,11111111
    # After reshape: byte 0: ch0/s0, ch0/s1, ch0/s2, ch0/s3, + mag.
    #                byte 1: ch1/s0, ch1/s1, ch1/s2, ch1/s3, + mag.
    frame = frame.view(np.uint8).reshape(-1, 2)
    # The look-up table splits each data word into the above 8 measurements,
    # the transpose pushes channels first and fanout last, and the reshape
    # flattens the fanout.
    return lut2bit3.take(frame, axis=0).transpose(1, 0, 2).reshape(2, -1).T


def encode_2chan_2bit_fanout4(values):
    """Encode payload for 2 channels using 2 bits, fan-out 4 (16 tracks)."""
    # Reverse reshaping (see above).
    values = values.reshape(-1, 4, 2).transpose(0, 2, 1)
    bitvalues = encode_2bit_base(values)
    # Values are -3, -1, +1, 3 -> 00, 01, 10, 11; get first bit (sign) as 1,
    # second bit (magnitude) as 16.
    reorder_bits = np.array([0, 16, 1, 17], dtype=np.uint8)
    reorder_bits.take(bitvalues, out=bitvalues)
    bitvalues <<= np.array([0, 1, 2, 3], dtype=np.uint8)
    out = np.bitwise_or.reduce(bitvalues, axis=-1).ravel().view('<u2')
    return out


def decode_4chan_2bit_fanout4(frame):
    """Decode payload for 4 channels using 2 bits, fan-out 4 (32 tracks)."""
    # Bitwise reordering of tracks, to align sign and magnitude bits,
    # reshaping to get VLBI channels in sequential, but wrong order.
    frame = reorder32(frame.view(np.uint32)).view(np.uint8).reshape(-1, 4)
    # Correct ordering.
    frame = frame.take(np.array([0, 2, 1, 3]), axis=1)
    # The look-up table splits each data byte into 4 measurements.
    # Using transpose ensures channels are first, then time samples, then
    # those 4 measurements, so the reshape orders the samples correctly.
    # Another transpose ensures samples are the first dimension.
    return lut2bit1.take(frame.T, axis=0).reshape(4, -1).T


def encode_4chan_2bit_fanout4(values):
    """Encode payload for 4 channels using 2 bits, fan-out 4 (32 tracks)."""
    reorder_channels = np.array([0, 2, 1, 3])
    values = values[:, reorder_channels].reshape(-1, 4, 4).transpose(0, 2, 1)
    bitvalues = encode_2bit_base(values)
    reorder_bits = np.array([0, 2, 1, 3], dtype=np.uint8)
    reorder_bits.take(bitvalues, out=bitvalues)
    bitvalues <<= np.array([0, 2, 4, 6], dtype=np.uint8)
    out = np.bitwise_or.reduce(bitvalues, axis=-1).ravel().view(np.uint32)
    return reorder32(out).view('<u4')


def decode_8chan_2bit_fanout2(frame):
    """Decode payload for 8 channels using 2 bits, fan-out 4 (32 tracks)."""
    # header['magnitude_bit'] = 00001111,00001111,00001111,00001111
    # makes sense with lut2bit3
    # header['fan_out'] = 00110011,00110011,00110011,00110011
    # i.e., s0s0,s1s1,m0m0,m1m1 for each byte
    # header['converter_id'] = 02020202,13131313,02020202,13131313
    # header['lsb_output'] =   00000000,00000000,11111111,11111111
    # After reshape: byte 0: ch0/s0, ch4/s0, ch0/s1, ch4/s1, + mag.
    #                byte 1: ch1/s0, ch5/s0, ch1/s1, ch5/s1, + mag.
    #                byte 2: ch2/s0, ch6/s0, ch2/s1, ch6/s1, + mag.
    #                byte 3: ch3/s0, ch7/s0, ch3/s1, ch7/s1, + mag.
    frame = frame.view(np.uint8).reshape(-1, 4)
    # The look-up table splits each data word into the above 16 measurements.
    # the first reshape means one gets time, channel&0x3, sample, channel&0x4
    # the transpose makes this channel&0x4, channel&0x3, time, sample.
    # the second reshape (which makes a copy) gets one just channel, time,
    # and the final transpose time, channel.
    return (lut2bit3.take(frame, axis=0).reshape(-1, 4, 2, 2)
            .transpose(3, 1, 0, 2).reshape(8, -1).T)


def encode_8chan_2bit_fanout2(values):
    """Encode payload for 8 channels using 2 bits, fan-out 2 (32 tracks)."""
    # words encode 16 values (see above) in order ch&0x3, sample, ch&0x4
    values = (values.reshape(-1, 2, 2, 4).transpose(0, 3, 1, 2)
              .reshape(-1, 4, 4))
    bitvalues = encode_2bit_base(values)
    # values are -3, -1, +1, 3 -> 00, 01, 10, 11;
    # get first bit (sign) as 1, second bit (magnitude) as 16
    reorder_bits = np.array([0, 16, 1, 17], dtype=np.uint8)
    reorder_bits.take(bitvalues, out=bitvalues)
    bitvalues <<= np.array([0, 1, 2, 3], dtype=np.uint8)
    out = np.bitwise_or.reduce(bitvalues, axis=-1).ravel().view('<u4')
    return out


def decode_16chan_2bit_fanout2_ft(frame):
    """Decode payload for 16 channels using 2 bits, fan-out 2 (64 tracks)."""
    # These Fortaleza files have an unusual ordering:
    # header['magnitude_bit'] = 00000101,10101111,00001111,00001111 * 2
    # White later two are as for lut2bit3, the first two are different.
    # header['fan_out'] = 00110011 * 8
    # i.e., s0s0,s1s1,m0m0,m1m1 for the later bytes.
    # header['converter_id'] = 03030303,04040404,15151515,26262626,
    #                          7a7a7a7a,7b7b7b7b,8c8c8c8c,9d9d9d9d
    # 0, 7 are doubled; those have lsb & usb;
    # header['lsb_output'] =   00001010,00001010,00000000,00000000 * 2
    # Should take magnitude bit literally,
    # byte 0: 0u/s0, 3u/s0, 0u/s1, 3u/s1, 0l/s0, 3u/m0, 0l/s1, 3u/m1
    # byte 1: 0u/m0, 4u/s0, 0u/m1, 4u/s1, 0l/m0, 4u/m0, 0l/m1, 4u/m1
    # byte 2: 1u/s0, 5u/s0, 1u/s1, 5u/s1, 1u/m0, 5u/m0, 1u/m1, 5u/m1
    # byte 3: 2u/s0, 6u/s0, 2u/s1, 6u/s1, 2u/m0, 6u/m0, 2u/m1, 6u/m1
    # byte 4: 7u/s0, au/s0, 7u/s1, au/s1, 7l/s0, au/m0, 7l/s1, au/m1
    # byte 5: 7u/m0, bu/s0, 7u/m1, bu/s1, 7l/m0, bu/m0, 7l/m1, bu/m1
    # byte 6: 8u/s0, cu/s0, 8u/s1, cu/s1, 8u/m0, cu/m0, 8u/m1, cu/m1
    # byte 7: 9u/s0, du/s0, 9u/s1, du/s1, 9u/m0, du/m0, 9u/m1, du/m1
    # This means the re-ordering is different from the usual: we just
    # need to shift bits around in bytes 0,1,4,5:
    frame = reorder64_Ft(frame.view(np.uint64))
    # This leaves samples as
    # byte 0: 0u/s0, 3u/s0, 0u/s1, 3u/s1, 0u/m0, 3u/m0, 0u/m1, 3u/m1
    # byte 1: 0l/s0, 4u/s0, 0l/s1, 4u/s1, 0l/m0, 4u/m0, 0l/m1, 4u/m1
    # byte 2: 1u/s0, 5u/s0, 1u/s1, 5u/s1, 1u/m0, 5u/m0, 1u/m1, 5u/m1
    # byte 3: 2u/s0, 6u/s0, 2u/s1, 6u/s1, 2u/m0, 6u/m0, 2u/m1, 6u/m1
    # byte 4: 7u/s0, au/s0, 7u/s1, au/s1, 7u/m0, au/m0, 7u/m1, au/m1
    # byte 5: 7l/s0, bu/s0, 7l/s1, bu/s1, 7l/m0, bu/m0, 7l/m1, bu/m1
    # byte 6: 8u/s0, cu/s0, 8u/s1, cu/s1, 8u/m0, cu/m0, 8u/m1, cu/m1
    # byte 7: 9u/s0, du/s0, 9u/s1, du/s1, 9u/m0, du/m0, 9u/m1, du/m1
    frame = frame.view(np.uint8).reshape(-1, 8)
    # The look-up table splits each data word into the above 32 measurements.
    # Translating 0u=0, 0l=1, 1..6=2..7, 7u=8, 7l=9, 8..d=a..f, the

    # first reshape yieds time, channel&0x8, channel&0x3, sample, channel&0x4
    # transpose makes this channel&0x8, channel&0x4, channel&0x3, time, sample.
    # and the second reshape (which makes a copy) gets one just time, channel,
    # and the final transpose time, channel.
    return (lut2bit3.take(frame, axis=0).reshape(-1, 2, 4, 2, 2)
            .transpose(1, 4, 2, 0, 3).reshape(16, -1).T)


def encode_16chan_2bit_fanout2_ft(values):
    """Encode payload for 16 channels using 2 bits, fan-out 2 (64 tracks)."""
    # words encode 32 values (above) in order ch&0x8, ch&0x3, sample, ch&0x4
    # First reshape goes to time, sample, ch&0x8, ch&0x4, ch&0x3,
    # transpose makes this time, ch&0x8, ch&0x3, sample, ch&0x4.
    # second reshape future bytes, sample+ch&0x4.
    values = (values.reshape(-1, 2, 2, 2, 4).transpose(0, 2, 4, 1, 3)
              .reshape(-1, 4))
    bitvalues = encode_2bit_base(values)
    # values are -3, -1, +1, 3 -> 00, 01, 10, 11;
    # get first bit (sign) as 1, second bit (magnitude) as 16
    reorder_bits = np.array([0, 16, 1, 17], dtype=np.uint8)
    reorder_bits.take(bitvalues, out=bitvalues)
    bitvalues <<= np.array([0, 1, 2, 3], dtype=np.uint8)
    out = np.bitwise_or.reduce(bitvalues, axis=-1).ravel().view(np.uint64)
    return reorder64_Ft(out).view('<u8')


def decode_8chan_2bit_fanout4(frame):
    """Decode payload for 8 channels using 2 bits, fan-out 4 (64 tracks)."""
    # Bitwise reordering of tracks, to align sign and magnitude bits,
    # reshaping to get VLBI channels in sequential, but wrong order.
    frame = reorder64(frame.view(np.uint64)).view(np.uint8).reshape(-1, 8)
    # Correct ordering.
    frame = frame.take(np.array([0, 2, 1, 3, 4, 6, 5, 7]), axis=1)
    # The look-up table splits each data byte into 4 measurements.
    # Using transpose ensures channels are first, then time samples, then
    # those 4 measurements, so the reshape orders the samples correctly.
    # Another transpose ensures samples are the first dimension.
    return lut2bit1.take(frame.T, axis=0).reshape(8, -1).T


def encode_8chan_2bit_fanout4(values):
    """Encode payload for 8 channels using 2 bits, fan-out 4 (64 tracks)."""
    reorder_channels = np.array([0, 2, 1, 3, 4, 6, 5, 7])
    values = values[:, reorder_channels].reshape(-1, 4, 8).transpose(0, 2, 1)
    bitvalues = encode_2bit_base(values)
    reorder_bits = np.array([0, 2, 1, 3], dtype=np.uint8)
    reorder_bits.take(bitvalues, out=bitvalues)
    bitvalues <<= np.array([0, 2, 4, 6], dtype=np.uint8)
    out = np.bitwise_or.reduce(bitvalues, axis=-1).ravel().view(np.uint64)
    return reorder64(out).view('<u8')


class Mark4Payload(PayloadBase):
    """Container for decoding and encoding Mark 4 payloads.

    Parameters
    ----------
    words : `~numpy.ndarray`
        Array containg LSB unsigned words (with the right size) that
        encode the payload.
    header : `~baseband.mark4.Mark4Header`, optional
        If given, used to infer the number of channels, bps, and fanout.
    sample_shape : tuple
        Shape of the samples (e.g., (nchan,)).  Default: (1,).
    bps : int, optional
        Number of bits per sample, used if ``header`` is not given.
        Default: 2.
    fanout : int, optional
        Number of tracks every bit stream is spread over, used if ``header`` is
        not given.  Default: 1.
    magnitude_bit : int, optional
        Magnitude bits for all tracks packed together. Used to index
        encoder and decoder.  Default: assume standard Mark 4 payload,
        for which number of channels, bps, and fanout suffice.

    Notes
    -----
    The total number of tracks is ``nchan * bps * fanout``.
    """

    _dtype_word = None
    # Decoders keyed by (nchan, nbit, fanout).
    _encoders = {(2, 2, 4): encode_2chan_2bit_fanout4,
                 (4, 2, 4): encode_4chan_2bit_fanout4,
                 (8, 2, 2): encode_8chan_2bit_fanout2,
                 (8, 2, 4): encode_8chan_2bit_fanout4,
                 (16, 0xf0faf050f0faf05, 2): encode_16chan_2bit_fanout2_ft}
    _decoders = {(2, 2, 4): decode_2chan_2bit_fanout4,
                 (4, 2, 4): decode_4chan_2bit_fanout4,
                 (8, 2, 2): decode_8chan_2bit_fanout2,
                 (8, 2, 4): decode_8chan_2bit_fanout4,
                 (16, 0xf0faf050f0faf05, 2): decode_16chan_2bit_fanout2_ft}

    _sample_shape_maker = namedtuple('SampleShape', 'nchan')

    def __init__(self, words, header=None, *, sample_shape=(1,), bps=2,
                 fanout=1, magnitude_bit=None, complex_data=False):
        if header is not None:
            magnitude_bit = header['magnitude_bit']
            bps = 2 if magnitude_bit.any() else 1
            ta = header.track_assignment
            if bps == 1 or np.all(magnitude_bit[ta] == [False, True]):
                magnitude_bit = None
            else:
                magnitude_bit = (np.packbits(magnitude_bit)
                                 .view(header.stream_dtype).item())

            ntrack = header.ntrack
            fanout = header.fanout
            sample_shape = (ntrack // (bps * fanout),)
            self._nbytes = header.payload_nbytes
        else:
            ntrack = sample_shape[0] * bps * fanout
            magnitude_bit = None

        self._dtype_word = MARK4_DTYPES[ntrack]
        self.fanout = fanout
        super().__init__(words, sample_shape=sample_shape,
                         bps=bps, complex_data=complex_data)
        self._coder = (self.sample_shape.nchan,
                       (self.bps if magnitude_bit is None else magnitude_bit),
                       self.fanout)

    @fixedvalue
    def complex_data(self):
        return False

    @classmethod
    def fromfile(cls, fh, header=None, **kwargs):
        """Read payload from filehandle and decode it into data.

        Parameters
        ----------
        fh : filehandle
            From which data is read.
        header : `~baseband.mark4.Mark4Header`
            Used to infer ``payload_nbytes``, ``bps``, ``sample_shape``, and
            ``dtype``.  If not given, those have to be passed in.
        """
        if header is not None:
            kwargs.setdefault('dtype', header.stream_dtype)
        return super().fromfile(fh, header=header, **kwargs)

    @classmethod
    def fromdata(cls, data, header):
        """Encode data as payload, using header information."""
        if data.dtype.kind == 'c':
            raise ValueError("Mark4 format does not support complex data.")
        if header.sample_shape != data.shape[1:]:
            raise ValueError("header is for {0} channels but data has {1}"
                             .format(header.nchan, data.shape[-1]))
        words = np.empty(header.payload_nbytes // header.stream_dtype.itemsize,
                         header.stream_dtype)
        self = cls(words, header)
        self[:] = data
        return self
