# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..vlbi_base.frame import VLBIFrameBase
from .header import GSBHeader
from .payload import GSBPayload


__all__ = ['GSBFrame']


class GSBFrame(VLBIFrameBase):
    """Frame encapsulating GSB rawdump or phased data.

    For rawdump data, lines in the timestamp file are associated with single
    blocks of raw data.  For phased data, the lines are associated with one or
    two polarisations, each consisting of two blocks of raw data. Hence, the
    raw data come from two or four files.

    Parameters
    ----------
    header : `~baseband.gsb.GSBHeader`
        Based on line from rawdump or phased timestamp file.
    payload : `~baseband.gsb.GSBPayload`
        Based on a single block of rawdump data, or the combined blocks for
        phased data.
    valid : bool, optional
        Whether the frame contains valid data.  Default: `True`.
    verify : bool, optional
        Whether to verify consistency of the frame parts.  Default: `True`.
    """
    _header_class = GSBHeader
    _payload_class = GSBPayload

    @classmethod
    def fromfile(cls, fh_ts, fh_raw, payloadsize=1 << 24, nchan=1, bps=4,
                 complex_data=False, valid=True, verify=True):
        """Read a frame from timestamp and raw data filehandles.

        Any arguments beyond the filehandle are used to help initialize the
        payload, except for ``valid`` and ``verify``, which are passed on
        to the header and class initializers.

        Parameters
        ----------
        fh_ts : filehandle
            To the timestamp file.  The next line will be read.
        fh_raw : file_handle or tuple
            Should be a single handle for a rawdump data frame, or a tuple
            containing tuples with pairs of handles for a phased one.  E.g.,
            ``((L1, L2), (R1, R2))`` for left and right polarisations.
        payloadsize : int, optional
            Size of the individual payloads.  Default: 4 MiB.
        nchan : int, optional
            Number of channels.  Default: 1.
        bps : int, optional
            Bits per elementary sample.  Default: 4.
        complex_data : bool, optional
            Whether data is complex.  Default: `False`.
        valid : bool, optional
            Whether the frame contains valid data.  Default: `True`.
        verify : bool, optional
            Whether to verify consistency of the frame parts.  Default: `True`.
        """
        header = cls._header_class.fromfile(fh_ts, verify=verify)
        payload = cls._payload_class.fromfile(fh_raw, payloadsize=payloadsize,
                                              nchan=nchan, bps=bps,
                                              complex_data=complex_data)
        return cls(header, payload, valid=valid, verify=verify)

    def tofile(self, fh_ts, fh_raw):
        """Write encoded frame to timestamp and raw data filehandles.

        Parameters
        ----------
        fh_ts : filehandle
            To the timestamp file.  A line will be added to it.
        fh_raw : file_handle or tuple
            Should be a single handle for a rawdump data frame, or a tuple
            containing tuples with pairs of handles for a phased one.  E.g.,
            ``((L1, L2), (R1, R2))`` for left and right polarisations.
        """
        self.header.tofile(fh_ts)
        self.payload.tofile(fh_raw)

    @property
    def size(self):
        """Size of the encoded frame in the raw data file in bytes."""
        return self.payload.size
