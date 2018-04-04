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
        Whether the data are valid.  Default: `True`.
    verify : bool, optional
        Whether to verify consistency of the frame parts.  Default: `True`.

    Notes
    -----
    GSB files do not support storing whether data are valid or not on disk.
    Hence, this has to be determined independently.  If ``valid=False``, any
    decoded data are set to ``cls.fill_value`` (by default, 0).

    The Frame can also be read instantiated using class methods:

      fromfile : read header and payload from their respective filehandles

      fromdata : encode data as payload

    Of course, one can also do the opposite:

      tofile : method to write header and payload to filehandles (splitting
               payload in the appropriate files).

      data : property that yields full decoded payload

    A number of properties are defined: `shape`, `dtype` and `size` are
    the shape, type and number of complete samples of the data array, and
    `nbytes` the frame size in bytes.  Furthermore, the frame acts as a
    dictionary, with keys those of the header.  Any attribute that is not
    defined on the frame itself, such as ``.time`` will be looked up on the
    header as well.
    """
    _header_class = GSBHeader
    _payload_class = GSBPayload

    @classmethod
    def fromfile(cls, fh_ts, fh_raw, payload_nbytes=1 << 24, nchan=1, bps=4,
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
        payload_nbytes : int, optional
            Size of the individual payloads in bytes.  Default: ``2**24``
            (16 MB).
        nchan : int, optional
            Number of channels.  Default: 1.
        bps : int, optional
            Bits per elementary sample.  Default: 4.
        complex_data : bool, optional
            Whether data are complex.  Default: `False`.
        valid : bool, optional
            Whether the data are valid (default: `True`). Note that this cannot
            be inferred from the header or payload itself.  If `False`, any
            data read will be set to ``cls.fill_value``.
        verify : bool, optional
            Whether to verify consistency of the frame parts.  Default: `True`.
        """
        header = cls._header_class.fromfile(fh_ts, verify=verify)
        payload = cls._payload_class.fromfile(fh_raw,
                                              payload_nbytes=payload_nbytes,
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
    def nbytes(self):
        """Size of the encoded frame in the raw data file in bytes."""
        return self.payload.nbytes
