# Licensed under the GPLv3 - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..vlbi_base.frame import VLBIFrameBase
from .header import GUPPIHeader
from .payload import GUPPIPayload


__all__ = ['GUPPIFrame']


class GUPPIFrame(VLBIFrameBase):
    """Representation of a GUPPI file, consisting of a header and payload.

    Parameters
    ----------
    header : `~baseband.dada.GUPPIHeader`
        Wrapper around the header lines, providing access to the values.
    payload : `~baseband.dada.GUPPIPayload`
        Wrapper around the payload, provding mechanisms to decode it.
    valid : bool
        Whether this frame contains valid data (default: True).
    verify : bool
        Whether to do basic verification of integrity (default: True)

    Notes
    -----

    GUPPI files do not support storing whether data are valid or not on disk.
    Hence, this has to be determined independently.  If ``valid=False``, any
    decoded data is set to ``cls.invalid_data_value`` (by default, 0).

    The Frame can also be instantiated using class methods:

      fromfile : read header and and map or read payload from a filehandle

      fromdata : encode data as payload

    Of course, one can also do the opposite:

      tofile : method to write header and payload to filehandle

      data : property that yields full decoded payload

    One can decode part of the payload by indexing or slicing the frame.

    A number of properties are defined: ``shape`` and ``dtype`` are the shape
    and type of the data array, and ``size`` the frame size in bytes.
    Furthermore, the frame acts as a dictionary, with keys those of the header.
    Any attribute that is not defined on the frame itself, such as ``.time``
    will be looked up on the header as well.
    """
    _header_class = GUPPIHeader
    _payload_class = GUPPIPayload

    @classmethod
    def fromfile(cls, fh, memmap=True, valid=True, verify=True):
        """Read a frame from a filehandle, possible mapping the payload.

        Parameters
        ----------
        fh : filehandle
            To read header from.
        memmap : bool, optional
            If `True` (default), use `~numpy.memmap` to map the payload.
            If `False`, just read it from disk.
        valid : bool, optional
            Whether the data is valid. Note that this cannot be inferred from
            the header or payload itself.  If `True`, any data read will be
            set to ``cls.invalid_data_value``.
        verify : bool, optional
            Whether to do basic verification of integrity.  Default: `True`.
        """
        header = cls._header_class.fromfile(fh, verify)
        payload = cls._payload_class.fromfile(fh, header=header, memmap=memmap)
        return cls(header, payload, valid=valid, verify=verify)

    @classmethod
    def fromdata(cls, data, header=None, valid=True, verify=True, **kwargs):
        """Construct frame from data and header.

        Note that since GUPPI files are often very large, one would normally
        map the file, and then set pieces of it by assigning to slices of the
        frame.  See `~baseband.base.GUPPIFileWriter.memmap_frame`.

        Parameters
        ----------
        data : ndarray
            Array holding complex or real data to be encoded.
        header : `~baseband.dada.GUPPIHeader` or None, optional
            If `None`, it will be attemtped to create one using the keywords.
        valid : bool, optional
            Whether the data is valid. Note that this information cannot be
            written to disk.
        verify : bool
            Whether or not to do basic assertions that check the integrity.
        **kwargs
            Used to construct a header if it was not explicitly passed in.
        """
        if header is None:
            header = cls._header_class.fromvalues(verify=verify, **kwargs)
        payload = cls._payload_class.fromdata(data, header.bps)
        return cls(header, payload, valid=valid, verify=verify)
