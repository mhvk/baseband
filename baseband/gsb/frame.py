# Licensed under the GPLv3 - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..vlbi_base.frame import VLBIFrameBase
from .header import GSBHeader
from .payload import GSBPayload


__all__ = ['GSBFrame']


class GSBFrame(VLBIFrameBase):
    """Frame encapsulating GSB Rawdump or Phased data.

    For Rawdump data, lines in the timestamp file are associated with single
    blocks of raw data.  For Phased data, the lines are associated with one or
    two polarisations, each consisting of two blocks of raw data. Hence, the
    raw data come from two or four files.

    Parameters
    ----------
    header : `~baseband.gsb.GSBHeader`
        Based on line from Rawdump or Phased timestamp file.
    payload : `~baseband.gsb.GSBPayload` or tuple
        Should be a single instance for a Rawdump data frame, or a tuple
        containing tuples with pairs of instances for a Phased data.
    valid : bool
        Whether the frame contains valid data (default: `True`).
    verify : bool
        Whether to verify consistency of the frame parts (default: `True`).
    """
    _header_class = GSBHeader
    _payload_class = None

    def __init__(self, header, payload, valid=True, verify=True):
        self._payload_class = gsb_payload_classes[header.mode]
        return super(GSBFrame, self).__init__(header, payload,
                                              valid=valid, verify=verify)

    def verify(self):
        """Verify that frame has right classes."""
        # This omits the superclass's check on header.payloadsize,
        # which cannot be done for GSB data.
        assert isinstance(self.header, self._header_class)
        assert isinstance(self.payload, self._payload_class)

    @classmethod
    def fromfile(cls, fh_ts, fh_raw, payloadsize=1 << 24, nchan=1, bps=4,
                 complex_data=False, valid=True, verify=True):
        """Read a frame from timestamp and raw data file handles.

        Any arguments beyond the filehandle are used to help initialize the
        payload, except for ``valid`` and ``verify``, which are passed on
        to the header and class initializers.

        Parameters
        ----------
        fh_ts : file handle
            To the timestamp file.  The next line will be read.
        fh_raw : file_handle or tuple
            Should be a single handle for a Rawdump data frame, or a tuple
            containing tuples with pairs of handles for a Phased data.  E.g.,
            ``((L1, L2), (R1, R2))`` for left and right polarisations.
        payloadsize : int
            Size of the individual payloads.  Default: 4 MiB.
        nchan : int
            Number of channels in the data.  Default: 1.
        bps : int
            Number of bits per sample part (i.e., per channel and per real or
            imaginary component).  Default: 4.
        complex_data : bool
            Whether data is complex or float.  Default: `False`.
        valid : bool
            Whether the frame contains valid data (default: `True`).
        verify : bool
            Whether to verify consistency of the frame parts (default: `True`).
        """
        header = cls._header_class.fromfile(fh_ts, verify=verify)
        payload_cls = gsb_payload_classes[header.mode]
        payload = payload_cls.fromfile(fh_raw, payloadsize=payloadsize,
                                       nchan=nchan, bps=bps,
                                       complex_data=complex_data)
        return cls(header, payload, valid=valid, verify=verify)

    def tofile(self, fh_ts, fh_raw):
        """Write encoded frame to timestamp and raw data file handles.

        Parameters
        ----------
        fh_ts : file handle
            To the timestamp file.  A line will be added to it.
        fh_raw : file_handle or tuple
            Should be a single handle for a Rawdump data frame, or a tuple
            containing tuples with pairs of handles for a Phased data, with the
            length of the tuple matching the number of threads.  E.g.,
            ``((L1, L2), (R1, R2))`` for data with left and right polarisation.
        """
        self.header.tofile(fh_ts)
        self.payload.tofile(fh_raw)

    @classmethod
    def fromdata(cls, data, header, bps=4, valid=True, verify=True):
        """Construct frame from data and header.

        Parameters
        ----------
        data : ndarray
            Array holding data to be encoded.
        header : `~baseband.gsb.GSBHeader`
            Header for the frame.
        bps : int
            Number of bits per sample part (i.e., per channel and per real or
            imaginary component).  Default: 4 for Rawdump, 8 for Phased.
        valid : bool
            Whether the frame contains valid data (default: `True`).
        verify : bool
            Whether to verify consistency of the frame parts (default: `True`).
        """
        payload_cls = gsb_payload_classes[header.mode]
        payload = payload_cls.fromdata(data, bps=bps)
        return cls(header, payload, valid=valid, verify=verify)

    @property
    def size(self):
        """Size of the encoded frame in the raw data file in bytes."""
        return self.payload.size


def _get_fh(fh):
    """Get a single filehandle out of a tuple (of tuples).

    Gets filehandle out of ``(fh,)`` or ``((fh,),)``.
    """
    while isinstance(fh, tuple):
        if len(fh) > 1:
            raise ValueError("A tuple containing filehandlers can "
                             "only contain a single item.")
        fh = fh[0]
    return fh


class GSBSinglePayload(GSBPayload):

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        return super(GSBSinglePayload,
                     cls).fromfile(_get_fh(fh), *args, **kwargs)

    def tofile(self, fh, *args, **kwargs):
        super(GSBSinglePayload, self).tofile(_get_fh(fh), *args, **kwargs)


class GSBPayloadSet(tuple):

    _data = None
    _payload_class = GSBPayload

    def verify(self):
        sizes = set()
        for pair in self:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            for pl in pair:
                assert isinstance(pl, self._payload_class)
                sizes.add(pl.size)
        assert len(sizes) == 1

    @classmethod
    def fromfile(cls, fh_raw, *args, **kwargs):
        self = cls(tuple(cls._payload_class.fromfile(fh, *args, **kwargs)
                         for fh in fh_pair)
                   for fh_pair in fh_raw)
        if kwargs.get('verify', True):
            self.verify()
        return self

    def tofile(self, fh_raw):
        for payload_pair, fh_pair in zip(self, fh_raw):
            for payload, fh in zip(payload_pair, fh_pair):
                payload.tofile(fh)

    @classmethod
    def fromdata(cls, data, *args, **kwargs):
        self = cls(tuple(cls._payload_class.fromdata(d, *args, **kwargs)
                         for d in thread.reshape((2, -1) + thread.shape[1:]))
                   for thread in data)
        if kwargs.get('verify', True):
            self.verify()
        return self

    def todata(self, data=None, invalid_data_value=0.):
        """Decode the payloads.

        Parameters
        ----------
        data : None or ndarray
            If given, the data is decoded into the array (which should have
            the correct shape).  By default, a new array is created.
        invalid_data_value : float
            Value to use for invalid data frames (default: 0.).
        """
        if data is None:
            if self._data is not None:
                return self._data

            data = np.empty(self.shape, dtype=self.dtype)

        for payload_pair, thread in zip(self, data):
            # Use assignment to shape rather than reshape to catch data
            # with wrong dimensions.
            thread.shape = 2, -1
            for payload, part in zip(payload_pair, thread):
                payload.todata(part)

        return data

    data = property(todata, doc="Decode and combine all payloads.")

    @property
    def size(self):
        """Size of the combined payloads (in bytes)."""
        return len(self) * 2 * self[0][0].size

    @property
    def shape(self):
        """Shape of the data encoded in the combined payloads."""
        pl0sh = self[0][0].shape
        return (len(self), 2 * pl0sh[0]) + pl0sh[1:]

    @property
    def dtype(self):
        """Type of the data encoded in the payloads."""
        return self[0][0].dtype


gsb_payload_classes = {'rawdump': GSBSinglePayload,
                       'phased': GSBPayloadSet}
