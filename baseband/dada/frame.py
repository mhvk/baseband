# Licensed under the GPLv3 - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..vlbi_base.frame import VLBIFrameBase
from .header import DADAHeader
from .payload import DADAPayload


__all__ = ['DADAFrame']


class DADAFrame(VLBIFrameBase):
    _header_class = DADAHeader
    _payload_class = DADAPayload

    @classmethod
    def fromfile(cls, fh, memmap=True, verify=True):
        """Read a frame from a filehandle, possible mapping the payload.

        Parameters
        ----------
        fh : filehandle
            To read header from.
        memmap : bool, optional
            If `True` (default), use `~numpy.memmap` to map the payload.
            If `False`, just read it from disk.
        verify : bool
            Whether to do basic verification of integrity.  Default: `True`.
        """
        header = cls._header_class.fromfile(fh, verify)
        payload = cls._payload_class.fromfile(fh, header=header, memmap=memmap)
        return cls(header, payload, verify=verify)

    @classmethod
    def fromdata(cls, data, header=None, verify=True, **kwargs):
        """Construct frame from data and header.

        Parameters
        ----------
        data : ndarray
            Array holding complex or real data to be encoded.
        header : `~baseband.dada.DADAHeader` or None, optional
            If `None`, it will be attemtped to create one using the keywords.
        verify : bool
            Whether or not to do basic assertions that check the integrity.
        """
        if header is None:
            header = cls._header_class.fromvalues(verify=verify, **kwargs)
        payload = cls._payload_class.fromdata(data, header.bps)
        return cls(header, payload, verify=verify)
