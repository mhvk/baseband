# Licensed under the GPLv3 - see LICENSE
"""Payload for GUPPI format."""
from collections import namedtuple

import numpy as np

from ..base.payload import PayloadBase


__all__ = ['GUPPIPayload']


def decode_8bit(words):
    return words.view(np.int8, np.ndarray).astype(np.float32)


def encode_8bit(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


class GUPPIPayload(PayloadBase):
    """Container for decoding and encoding GUPPI payloads.

    Parameters
    ----------
    words : `~numpy.ndarray`
        Array containg LSB unsigned words (with the right size) that
        encode the payload.
    header : `~baseband.guppi.GUPPIHeader`
        Header that provides information about how the payload is encoded.
        If not given, the following arguments have to be passed in.
    bps : int, optional
        Number of bits per sample part (i.e., per channel and per real or
        imaginary component).  Default: 8.
    sample_shape : tuple, optional
        Shape of the samples; e.g., (nchan,).  Default: ().
    complex_data : bool, optional
        Whether data are complex.  Default: `False`.
    channels_first : bool, optional
        Whether the encoded payload is stored as (nchan, nsample, npol),
        rather than (nsample, nchan, npol).  Default: `True`.
    """
    _decoders = {
        8: decode_8bit}
    _encoders = {
        8: encode_8bit}
    _dtype_word = np.dtype('int8')
    _memmap = True

    _sample_shape_maker = namedtuple('SampleShape', 'npol, nchan')

    def __init__(self, words, *, header=None, sample_shape=(), bps=8,
                 complex_data=False, channels_first=True):
        super().__init__(words, header=header, sample_shape=sample_shape,
                         bps=bps, complex_data=complex_data)
        self.channels_first = (channels_first if header is None
                               else header.channels_first)
        # If channels first, _item_to_slices must act on per-channel words.  By
        # resetting self._bpfs, we allow _item_to_slices to work unmodified.
        self._true_bpfs = self._bpfs    # Save the true bpfs regardless.
        if self.channels_first:
            self._bpfs //= self.sample_shape.nchan

    @classmethod
    def fromdata(cls, data, header=None, bps=8, channels_first=True):
        """Encode data as a payload.

        Parameters
        ----------
        data : `~numpy.ndarray`
            Data to be encoded. The trailing dimensions are taken as the
            sample shape, normally (npol, nchan).
        header : `~baseband.guppi.GUPPIHeader`, optional
            If given, used to infer the ``bps`` and ``channels_first``.
        bps : int, optional
            Bits per elementary sample, used only if ``header`` is `None`.
            Default: 8.
        channels_first : bool, optional
            If `True`, encode data (nchan, nsample, npol). otherwise
            as (nsample, nchan, npol).  Used only if ``header`` is `None`.
            Default: `True`.
        """
        return super().fromdata(data, header=header, bps=bps,
                                channels_first=channels_first)

    def __len__(self):
        """Number of samples in the payload."""
        return self.nbytes * 8 // self._true_bpfs

    def _decode(self, words, words_slice):
        if self.channels_first:
            # Before decoding, reshape so channels fall along first axis.
            decoded_words = super()._decode(
                words.reshape(self.sample_shape.nchan, -1)[:, words_slice])
            # Reshape to (nsample, nchan, npol), as expected by data_slice.
            return decoded_words.T.reshape(-1, *self.sample_shape)
        else:
            # Transpose result to allow data_slice to assume (npol, nchan).
            return (super()._decode(words[words_slice])
                    .reshape(-1, self.sample_shape.nchan,
                             self.sample_shape.npol)
                    .transpose(0, 2, 1))

    def __getitem__(self, item=()):
        # GUPPI data may be stored as (nsample, nchan, npol) or, if
        # channels-first, (nchan, nsample, npol), both of which require
        # reshaping to get the usual order of (nsample, npol, nchan).

        words_slice, data_slice = self._item_to_slices(item)
        return self._decode(self.words, words_slice)[data_slice]

    def __setitem__(self, item, data):
        words_slice, data_slice = self._item_to_slices(item)

        data = np.asanyarray(data)
        # Check if the new data spans an entire word and is correctly shaped.
        # If so, skip decoding.  If not, decode appropriate words and insert
        # new data.
        if not (data_slice == (slice(None),)
                and data.shape[-2:] == self.sample_shape
                and data.dtype.kind == self.dtype.kind):
            current_data = self._decode(self.words, words_slice)
            current_data[data_slice] = data
            data = current_data

        # Reshape before separating real and complex components.
        if self.channels_first:
            encoded_words = (self._encode(data.transpose(2, 0, 1))
                             .reshape(self.sample_shape.nchan, -1)
                             .view(self._dtype_word))
            self.words.reshape(self.sample_shape.nchan, -1)[:, words_slice] = (
                encoded_words)
        else:
            encoded_words = (self._encode(data.transpose(0, 2, 1))
                             .ravel().view(self._dtype_word))
            self.words[words_slice] = encoded_words

    data = property(__getitem__, doc="Full decoded payload.")
