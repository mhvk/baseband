# Licensed under the GPLv3 - see LICENSE
"""Payload for GUPPI format."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from collections import namedtuple

from ..vlbi_base.payload import VLBIPayloadBase


__all__ = ['GUPPIPayload']


def decode_8bit(words):
    return words.view(np.int8, np.ndarray).astype(np.float32)


def encode_8bit(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


class GUPPIPayload(VLBIPayloadBase):
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

    _sample_shape_maker = namedtuple('SampleShape', 'npol, nchan')

    def __init__(self, words, header=None, sample_shape=(), bps=8,
                 complex_data=False, channels_first=True):
        if header is not None:
            bps = header.bps
            sample_shape = header.sample_shape
            complex_data = header.complex_data
            channels_first = header.channels_first
        super(GUPPIPayload, self).__init__(words, sample_shape=sample_shape,
                                           bps=bps, complex_data=complex_data)
        self.channels_first = channels_first
        # If channels first, _item_to_slices must act on per-channel words.  By
        # resetting self._bpfs, we allow _item_to_slices to work unmodified.
        self._true_bpfs = self._bpfs    # Save the true bpfs regardless.
        if self.channels_first:
            self._bpfs //= self.sample_shape.nchan

    @classmethod
    def fromfile(cls, fh, header=None, memmap=False, payload_nbytes=None,
                 **kwargs):
        """Read or map encoded data in file.

        Parameters
        ----------
        fh : filehandle
            Handle to the file which will be read or mapped.
        header : `~baseband.guppi.GUPPIHeader`, optional
            If given, used to infer ``payload_nbytes``, ``bps``,
            ``sample_shape``, ``complex_data`` and ``channels_first``.  If not
            given, those have to be passed in.
        memmap : bool, optional
            If `False` (default), read from file.  Otherwise, map the file in
            memory (see `~numpy.memmap`).
        payload_nbytes : int, optional
            Number of bytes to read (default: as given in ``header``,
            ``cls._nbytes``, or, for mapping, to the end of the file).
        **kwargs
            Additional arguments are passed on to the class initializer. These
            are only needed if ``header`` is not given.
        """
        if payload_nbytes is None:
            payload_nbytes = (cls._nbytes if header is None
                              else header.payload_nbytes)

        if not memmap:
            return super(GUPPIPayload, cls).fromfile(
                fh, header=header, payload_nbytes=payload_nbytes, **kwargs)

        words_shape = (payload_nbytes // cls._dtype_word.itemsize,)
        if hasattr(fh, 'memmap'):
            words = fh.memmap(dtype=cls._dtype_word,
                              shape=(None if payload_nbytes is None else
                                     words_shape))
        else:
            mode = fh.mode.replace('b', '')
            offset = fh.tell()
            words = np.memmap(
                fh, mode=mode, dtype=cls._dtype_word, offset=offset,
                shape=(None if payload_nbytes is None else words_shape))
            fh.seek(offset + words.size * words.dtype.itemsize)
        return cls(words, header=header, **kwargs)

    @classmethod
    def fromdata(cls, data, header=None, bps=8, channels_first=True):
        """Encode data as a payload.

        Parameters
        ----------
        data : `~numpy.ndarray`
            Data to be encoded. The last dimension is taken as the number of
            channels.
        header : `~baseband.guppi.GUPPIHeader`, optional
            If given, used to infer the ``bps`` and ``channels_first``.
        bps : int, optional
            Bits per elementary sample, used if ``header`` is `None`.
            Default: 8.
        channels_first : bool, optional
            Whether encoded data should be ordered as (nchan, nsample, npol),
            used if ``header`` is `None`.  Default: `True`.
        """
        if header is not None:
            bps = header.bps
            channels_first = header.channels_first
        sample_shape = data.shape[1:]
        complex_data = data.dtype.kind == 'c'
        try:
            encoder = cls._encoders[bps]
        except KeyError:
            raise ValueError("{0} cannot encode data with {1} bits"
                             .format(cls.__name__, bps))
        # If channels-first, switch to (nchan, nsample, npol); otherwise use
        # (nsample, nchan, npol).
        if channels_first:
            data = data.transpose(2, 0, 1)
        else:
            data = data.transpose(0, 2, 1)
        if complex_data:
            data = data.view((data.real.dtype, (2,)))
        words = encoder(data).ravel().view(cls._dtype_word)
        return cls(words, sample_shape=sample_shape, bps=bps,
                   complex_data=complex_data, channels_first=channels_first)

    def __len__(self):
        """Number of samples in the payload."""
        return self.nbytes * 8 // self._true_bpfs

    def __getitem__(self, item=()):
        # GUPPI data may be stored as (nsample, nchan, npol) or, if
        # channels-first, (nchan, nsample, npol), both of which require
        # reshaping to get the usual order of (nsample, npol, nchan).
        decoder = self._decoders[self._coder]

        # If we want to decode the entire dataset.
        if item is () or item == slice(None):
            data = decoder(self.words)
            if self.complex_data:
                data = data.view(self.dtype)
            if self.channels_first:
                # Reshape to (nchan, nsample, npol); transpose to usual order.
                return (data.reshape(self.sample_shape.nchan, -1,
                                     self.sample_shape.npol)
                        .transpose(1, 2, 0))
            else:
                # Reshape to (nsample, nchan, npol); transpose to usual order.
                return (data.reshape(-1, self.sample_shape.nchan,
                                     self.sample_shape.npol)
                        .transpose(0, 2, 1))

        words_slice, data_slice = self._item_to_slices(item)

        if self.channels_first:
            # Reshape words so channels fall along first axis, then decode.
            decoded_words = decoder(
                self.words.reshape(self.sample_shape.nchan, -1)[:, words_slice])
            # Reshape to (nsample, nchan, npol), then use data_slice.
            return (decoded_words.view(self.dtype).T
                    .reshape(-1, *self.sample_shape)[data_slice])
        else:
            # data_slice assumes (npol, nchan), so transpose before using it.
            return (decoder(self.words[words_slice]).view(self.dtype)
                    .reshape(-1, self.sample_shape.nchan,
                             self.sample_shape.npol)
                    .transpose(0, 2, 1)[data_slice])

    def __setitem__(self, item, data):
        if item is () or item == slice(None):
            words_slice = data_slice = slice(None)
        else:
            words_slice, data_slice = self._item_to_slices(item)

        data = np.asanyarray(data)
        # Check if the new data spans an entire word and is correctly shaped.
        # If so, skip decoding.  If not, decode appropriate words and insert
        # new data.
        if not (data_slice == slice(None) and
                data.shape[-2:] == self.sample_shape and
                data.dtype.kind == self.dtype.kind):
            decoder = self._decoders[self._coder]
            if self.channels_first:
                decoded_words = decoder(np.ascontiguousarray(
                    self.words.reshape(
                        self.sample_shape.nchan, -1)[:, words_slice]))
                current_data = (decoded_words.view(self.dtype)
                                .T.reshape(-1, *self.sample_shape))
            else:
                current_data = (decoder(self.words[words_slice])
                                .view(self.dtype)
                                .reshape(-1, self.sample_shape.nchan,
                                         self.sample_shape.npol)
                                .transpose(0, 2, 1))

            current_data[data_slice] = data
            data = current_data

        # Reshape before separating real and complex components.
        if self.channels_first:
            data = data.reshape(-1, self.sample_shape.nchan).T
        else:
            data = data.transpose(0, 2, 1)

        # Separate real and complex components.
        if data.dtype.kind == 'c':
            data = data.view((data.real.dtype, (2,)))

        # Select encoder.
        encoder = self._encoders[self._coder]

        # Reshape and encode words.
        if self.channels_first:
            self.words.reshape(self.sample_shape.nchan, -1)[:, words_slice] = (
                encoder(data).reshape(self.sample_shape.nchan, -1)
                .view(self._dtype_word))
        else:
            self.words[words_slice] = (encoder(data.ravel())
                                       .view(self._dtype_word))

    data = property(__getitem__, doc="Full decoded payload.")
