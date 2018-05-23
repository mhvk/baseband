# Licensed under the GPLv3 - see LICENSE
"""
Base definitions for VLBI payloads, used for VDIF and Mark 5B.

Defines a payload class VLBIPayloadBase that can be used to hold the words
corresponding to a frame payload, providing access to the values encoded in
it as a numpy array.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import operator
from functools import reduce
import numpy as np


__all__ = ['VLBIPayloadBase']


class VLBIPayloadBase(object):
    """Container for decoding and encoding VLBI payloads.

    Any subclass should define dictionaries ``_decoders`` and ``_encoders``,
    which hold functions that decode/encode the payload words to/from ndarray.
    These dictionaries are assumed to be indexed by ``bps``.

    Parameters
    ----------
    words : `~numpy.ndarray`
        Array containg LSB unsigned words (with the right size) that
        encode the payload.
    sample_shape : tuple
        Shape of the samples (e.g., (nchan,)).  Default: ().
    bps : int
        Bits per elementary sample, i.e., per channel and per real or
        imaginary component.  Default: 2.
    complex_data : bool
        Whether the data are complex.  Default: `False`.
    """
    # Possible fixed payload size in bytes.
    _nbytes = None
    # Default type for encoded data.
    _dtype_word = np.dtype('<u4')
    """Default for words: 32-bit unsigned integers, with lsb first."""
    # To be defined by subclasses.
    _encoders = {}
    _decoders = {}
    # Placeholder for sample shape named tuple.
    _sample_shape_maker = None

    def __init__(self, words, sample_shape=(), bps=2, complex_data=False):
        self.words = words
        if self._sample_shape_maker is not None:
            self.sample_shape = self._sample_shape_maker(*sample_shape)
        else:
            self.sample_shape = sample_shape
        self.bps = bps
        self.complex_data = complex_data
        self._bpfs = (bps * (2 if complex_data else 1) *
                      reduce(operator.mul, sample_shape, 1))
        self._coder = bps
        if self._nbytes is not None and self._nbytes != self.nbytes:
            raise ValueError("encoded data should have length {0}"
                             .format(self._nbytes))
        if words.dtype != self._dtype_word:
            raise ValueError("encoded data should have dtype {0}"
                             .format(self._dtype_word))

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        """Read payload from filehandle and decode it into data.

        Parameters
        ----------
        fh : filehandle
            From which data is read.
        payload_nbytes : int
            Number of bytes to read (default: as given in ``cls._nbytes``).

        Any other (keyword) arguments are passed on to the class initialiser.
        """
        payload_nbytes = kwargs.pop('payload_nbytes', cls._nbytes)
        if payload_nbytes is None:
            raise ValueError("payload_nbytes should be given as an argument "
                             "if no default is defined on the class.")
        s = fh.read(payload_nbytes)
        if len(s) < payload_nbytes:
            raise EOFError("could not read full payload.")
        return cls(np.frombuffer(s, dtype=cls._dtype_word), *args, **kwargs)

    def tofile(self, fh):
        """Write payload to filehandle."""
        return fh.write(self.words.tostring())

    @classmethod
    def fromdata(cls, data, header=None, bps=2):
        """Encode data as a payload.

        Parameters
        ----------
        data : `~numpy.ndarray`
            Data to be encoded. The last dimension is taken as the number of
            channels.
        header : header instance, optional
            If given, used to infer the bps.
        bps : int, optional
            Bits per elementary sample, i.e., per channel and per real or
            imaginary component, used if header is not given.  Default: 2.
        """
        sample_shape = data.shape[1:]
        complex_data = data.dtype.kind == 'c'
        if header:
            bps = header.bps
        try:
            encoder = cls._encoders[bps]
        except KeyError:
            raise ValueError("{0} cannot encode data with {1} bits"
                             .format(cls.__name__, bps))
        if complex_data:
            data = data.view((data.real.dtype, (2,)))
        words = encoder(data).ravel().view(cls._dtype_word)
        return cls(words, sample_shape=sample_shape, bps=bps,
                   complex_data=complex_data)

    def __array__(self, dtype=None):
        """Interface to arrays."""
        if dtype is None or dtype == self.dtype:
            return self.data
        else:
            return self.data.astype(dtype)

    @property
    def nbytes(self):
        """Size of the payload in bytes."""
        return self.words.size * self.words.dtype.itemsize

    def __len__(self):
        """Number of samples in the payload."""
        return self.nbytes * 8 // self._bpfs

    @property
    def shape(self):
        """Shape of the decoded data array."""
        return (len(self),) + self.sample_shape

    @property
    def size(self):
        """Total number of component samples in the decoded data array."""
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod

    @property
    def ndim(self):
        """Number of dimensions of the decoded data array."""
        return len(self.shape)

    @property
    def dtype(self):
        """Numeric type of the decoded data array."""
        return np.dtype(np.complex64 if self.complex_data else np.float32)

    def _item_to_slices(self, item):
        """Get word and data slices required to obtain given item.

        Parameters
        ----------
        item : int, slice, or tuple
            Sample indices.  An int represents a single sample, a slice
            a sample range, and a tuple of ints/slices a range for
            multi-channel data.

        Returns
        -------
        words_slice : slice
            Slice such that if one decodes ``ds = self.words[words_slice]``,
            ``ds`` is the smallest possible array that includes all
            of the requested ``item``.
        data_slice : int or slice
            Int or slice such that ``decode(ds)[data_slice]`` is the requested
            ``item``.

        Notes
        -----
        ``item`` is restricted to (tuples of) ints or slices, so one cannot
        access non-contiguous samples using advanced indexing.  If ``item``
        is a slice, a negative increment cannot be used.  The function is
        unable to parse payloads whose words have unused space (eg. VDIF files
        with 20 bits/sample).
        """
        if isinstance(item, tuple):
            sample_index = item[1:]
            item = item[0]
        else:
            sample_index = ()

        nsample = len(self)
        is_slice = isinstance(item, slice)
        if is_slice:
            start, stop, step = item.indices(nsample)
            assert step > 0, "cannot deal with negative steps yet."
            n = stop - start
            if step == 1:
                step = None
        else:
            try:
                item = operator.index(item)
            except Exception:
                raise TypeError("{0} object can only be indexed or sliced."
                                .format(type(self)))
            if item < 0:
                item += nsample

            if not (0 <= item < nsample):
                raise IndexError("{0} index out of range.".format(type(self)))

            start, stop, step, n = item, item+1, 1, 1

        if n == nsample:
            words_slice = slice(None)
            data_slice = slice(None, None, step) if is_slice else 0

        else:
            bpw = 8 * self.words.dtype.itemsize
            bpfs = self._bpfs
            if bpfs % bpw == 0:
                # Each full sample requires one or more encoded words.
                # Get corresponding range in words required, and decode those.
                wpfs = bpfs // bpw
                words_slice = slice(start * wpfs, stop * wpfs)
                data_slice = slice(None, None, step) if is_slice else 0

            elif bpw % bpfs == 0:
                # Each word contains multiple samples.
                # Get words in which required samples are contained.
                fspw = bpw // bpfs
                w_start, o_start = divmod(start, fspw)
                w_stop, o_stop = divmod(stop, fspw)

                words_slice = slice(w_start, w_stop + 1 if o_stop else w_stop)
                data_slice = slice(o_start if o_start else None,
                                   o_start + n if o_stop else None,
                                   step) if is_slice else o_start

            else:
                raise TypeError("do not know how to extract data when full "
                                "samples have {0} bits and words have {1} bits"
                                .format(bpfs, bpw))

        return words_slice, (data_slice,) + sample_index

    def __getitem__(self, item=()):
        decoder = self._decoders[self._coder]
        if item is () or item == slice(None):
            data = decoder(self.words)
            if self.complex_data:
                data = data.view(self.dtype)
            return data.reshape(self.shape)

        words_slice, data_slice = self._item_to_slices(item)

        return (decoder(self.words[words_slice]).view(self.dtype)
                .reshape(-1, *self.sample_shape)[data_slice])

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
                data.shape[-len(self.sample_shape):] == self.sample_shape and
                data.dtype.kind == self.dtype.kind):
            decoder = self._decoders[self._coder]
            current_data = decoder(self.words[words_slice])
            if self.complex_data:
                current_data = current_data.view(self.dtype)
            current_data.shape = (-1,) + self.sample_shape
            current_data[data_slice] = data
            data = current_data

        if data.dtype.kind == 'c':
            data = data.view((data.real.dtype, (2,)))

        encoder = self._encoders[self._coder]
        self.words[words_slice] = encoder(data).ravel().view(self._dtype_word)

    data = property(__getitem__, doc="Full decoded payload.")

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.shape == other.shape and
                self.dtype == other.dtype and
                (self.words is other.words or
                 np.all(self.words == other.words)))

    def __ne__(self, other):
        return not self.__eq__(other)
