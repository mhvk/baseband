# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..vlbi_base.frame import VLBIFrameBase
from .header import DADAHeader
from .payload import DADAPayload


__all__ = ['DADAFrame']


class DADAFrame(VLBIFrameBase):
    """Representation of a DADA file, consisting of a header and payload.

    Parameters
    ----------
    header : `~baseband.dada.DADAHeader`
        Wrapper around the header lines, providing access to the values.
    payload : `~baseband.dada.DADAPayload`
        Wrapper around the payload, provding mechanisms to decode it.
    valid : bool, optional
        Whether the data are valid.  Default: `True`.
    verify : bool, optional
        Whether to do basic verification of integrity.  Default: `True`.

    Notes
    -----
    DADA files do not support storing whether data are valid or not on disk.
    Hence, this has to be determined independently.  If ``valid=False``, any
    decoded data are set to ``cls.fill_value`` (by default, 0).

    The Frame can also be instantiated using class methods:

      fromfile : read header and map or read payload from a filehandle

      fromdata : encode data as payload

    Of course, one can also do the opposite:

      tofile : method to write header and payload to filehandle

      data : property that yields full decoded payload

    One can decode part of the payload by indexing or slicing the frame.

    A number of properties are defined: `shape`, `dtype` and `size` are
    the shape, type and number of complete samples of the data array, and
    `nbytes` the frame size in bytes.  Furthermore, the frame acts as a
    dictionary, with keys those of the header.  Any attribute that is not
    defined on the frame itself, such as ``.time`` will be looked up on the
    header as well.
    """
    _header_class = DADAHeader
    _payload_class = DADAPayload

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
            Whether the data are valid (default: `True`). Note that this cannot
            be inferred from the header or payload itself.  If `False`, any
            data read will be set to ``cls.fill_value``.
        verify : bool, optional
            Whether to do basic verification of integrity.  Default: `True`.
        """
        header = cls._header_class.fromfile(fh, verify=verify)
        payload = cls._payload_class.fromfile(fh, header=header, memmap=memmap)
        return cls(header, payload, valid=valid, verify=verify)

    @classmethod
    def fromdata(cls, data, header=None, valid=True, verify=True, **kwargs):
        """Construct frame from data and header.

        Note that since DADA files are generally very large, one would normally
        map the file, and then set pieces of it by assigning to slices of the
        frame.  See `~baseband.dada.base.DADAFileWriter.memmap_frame`.

        Parameters
        ----------
        data : `~numpy.ndarray`
            Array holding complex or real data to be encoded.
        header : `~baseband.dada.DADAHeader` or None
            If not given, will attempt to generate one using the keywords.
        valid : bool, optional
            Whether the data are valid (default: `True`). Note that this
            information cannot be written to disk.
        verify : bool, optional
            Whether or not to do basic assertions that check the integrity.
            Default: `True`.
        **kwargs
            If ``header`` is not given, these are used to initialize one.
        """
        if header is None:
            header = cls._header_class.fromvalues(verify=verify, **kwargs)
        payload = cls._payload_class.fromdata(data, header=header)
        return cls(header, payload, valid=valid, verify=verify)
