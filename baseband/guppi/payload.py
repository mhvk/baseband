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
    header : `~baseband.dada.GUPPIHeader`
        Header that provides information about how the payload is encoded.
        If not given, the following arguments have to be passed in.
    bps : int, optional
        Number of bits per sample part (i.e., per channel and per real or
        imaginary component).  Default: 8.
    sample_shape : tuple, optional
        Shape of the samples; e.g., (nchan,).  Default: ().
    complex_data : bool, optional
        Whether data are complex.  Default: `False`.
    time_ordered : bool, optional
        Whether data are time-ordered.  Default: `True`.
    """
    _decoders = {
        8: decode_8bit}
    _encoders = {
        8: encode_8bit}

    _sample_shape_maker = namedtuple('SampleShape', 'npol, nchan')

    def __init__(self, words, header=None, sample_shape=(), bps=8,
                 complex_data=False, time_ordered=True):
        if header is not None:
            bps = header.bps
            sample_shape = header.sample_shape
            complex_data = header.complex_data
            time_ordered = header.time_ordered
        super(GUPPIPayload, self).__init__(words, sample_shape=sample_shape,
                                           bps=bps, complex_data=complex_data)
        self.time_ordered = time_ordered

    @classmethod
    def fromfile(cls, fh, header=None, memmap=False, payload_nbytes=None,
                 **kwargs):
        """Read or map encoded data in file.

        Parameters
        ----------
        fh : filehandle
            Handle to the file which will be read or mapped.
        header : `~baseband.dada.GUPPIHeader`, optional
            If given, used to infer ``payload_nbytes``, ``bps``,
            ``sample_shape``, ``complex_data`` and ``time_ordered``.  If not
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

        if hasattr(fh, 'memmap'):
            words = fh.memmap(dtype=cls._dtype_word,
                              shape=None if payload_nbytes is None else
                              (payload_nbytes // cls._dtype_word.itemsize,))
        else:
            mode = fh.mode.replace('b', '')
            offset = fh.tell()
            words = np.memmap(
                fh, mode=mode, dtype=cls._dtype_word, offset=offset,
                shape=(None if payload_nbytes is None
                       else (payload_nbytes // cls._dtype_word.itemsize,)))
            fh.seek(offset + words.size * words.dtype.itemsize)
        return cls(words, header=header, **kwargs)

    @classmethod
    def fromdata(cls, data, header=None, bps=8, time_ordered=True):
        """Encode data as a payload.

        Parameters
        ----------
        data : `~numpy.ndarray`
            Data to be encoded. The last dimension is taken as the number of
            channels.
        header : `~baseband.guppi.GUPPIHeader`, optional
            If given, used to infer the ``bps`` and ``time_ordered``.
        bps : int, optional
            Bits per elementary sample, used if ``header`` is `None`.
            Default: 8.
        time_ordered : bool, optional
            Whether data are time-ordered, used if ``header`` is `None`.
            Default: `True`.
        """
        if header is not None:
            bps = header.bps
            time_ordered = header.time_ordered
        sample_shape = data.shape[1:]
        complex_data = data.dtype.kind == 'c'
        try:
            encoder = cls._encoders[bps]
        except KeyError:
            raise ValueError("{0} cannot encode data with {1} bits"
                             .format(cls.__name__, bps))
        data = data.transpose(0, 2, 1)    # Switch to (nchan, npol)
        if time_ordered:                  # If time-ordered, channels go first.
            data = data.transpose(1, 0, 2)
        if complex_data:
            data = data.view((data.real.dtype, (2,)))
        words = encoder(data).ravel().view(cls._dtype_word)
        return cls(words, sample_shape=sample_shape, bps=bps,
                   complex_data=complex_data, time_ordered=time_ordered)

    def __getitem__(self, item=()):
        #  GUPPI data may be stored as (nsample, nchan, npol) or, if
        # "time-ordered", (nchan, nsample, npol), both of which require
        # reshaping to get the usual order of (nsample, npol, nchan).
        decoder = self._decoders[self._coder]

        # If we want to decode the entire dataset.
        if item is () or item == slice(None):
            data = decoder(self.words)
            if self.complex_data:
                data = data.view(self.dtype)
            if self.time_ordered:
                # Reshape to (nchan, nsample, npol); transpose to usual order.
                return (data.reshape(self.shape[2], self.shape[0],
                                     self.shape[1])
                        .transpose(1, 2, 0))
            else:
                # Reshape to (nsample, nchan, npol); transpose to usual order.
                return (data.reshape(self.shape[0], self.shape[2],
                                     self.shape[1])
                        .transpose(0, 2, 1))

        if self.time_ordered:
            data = self.__getitem__()  # Grab everything.
            return data.__getitem__(item)  # Subset via the array.
        else:
            #words_slice, data_slice = (
            #    super(GUPPIPayload, self)._item_to_slices(item))
            words_slice, data_slice = self._item_to_slices(item)
            data = (decoder(self.words[words_slice]).view(self.dtype)
                    .reshape(-1, self.sample_shape[1],
                             self.sample_shape[0])[data_slice])
            return data.transpose(0, 2, 1)

    data = property(__getitem__, doc="Full decoded payload.")

    def __setitem__(self, item, data):
        # Utterly thoughtless: decode entire payload, modify it, then encode
        # again.
        if not self.time_ordered:
            raise NotImplementedError
        data = np.asanyarray(data)
        data_full = self.__getitem__()  # Grab everything.
        data_full[item] = data          # Set values into decoded data.
        # Flip back to (nchan, nsample, npol).
        data_full = data_full.transpose(2, 0, 1)
        if data_full.dtype.kind == 'c':
            data_full = data_full.view((data_full.real.dtype, (2,)))
        encoder = self._encoders[self._coder]
        # [:] for memmaps.
        self.words[:] = encoder(data_full).ravel().view(self._dtype_word)
