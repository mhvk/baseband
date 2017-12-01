# Licensed under the GPLv3 - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

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
    payload : `~baseband.gsb.GSBPayload` or `~baseband.gsb.GSBPhasedPayload`
        Based on a single block of rawdump data, or the combined blocks for
        phased data.
    valid : bool
        Whether the frame contains valid data (default: `True`).
    verify : bool
        Whether to verify consistency of the frame parts (default: `True`).
    """
    _header_class = GSBHeader
    _payload_class = GSBPayload

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
            containing tuples with pairs of handles for phased data.  E.g.,
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
        payload = cls._payload_class.fromfile(fh_raw, payloadsize=payloadsize,
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

    @property
    def size(self):
        """Size of the encoded frame in the raw data file in bytes."""
        return self.payload.size
