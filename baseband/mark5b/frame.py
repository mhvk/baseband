from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..vlbi_base import VLBIFrameBase
from .header import Mark5BHeader
from .payload import Mark5BPayload


class Mark5BFrame(VLBIFrameBase):

    _header_class = Mark5BHeader
    _payload_class = Mark5BPayload
    _fill_pattern = 0x11223344

    def __init__(self, header, payload, valid=None, verify=True):
        if valid is None:
            # Is this payload OK?  Usually yes, so short-circuit on first few.
            valid = (payload.words[0] != self._fill_pattern or
                     payload.words[1] != self._fill_pattern or
                     payload.words[2] != self._fill_pattern or
                     (payload.words[3:] != self._fill_pattern).any())

        super(Mark5BFrame, self).__init__(header, payload, valid, verify)

    @classmethod
    def fromfile(cls, fh, ref_mjd, nchan, bps=2, valid=None, verify=True):
        header = cls._header_class.fromfile(fh, ref_mjd, verify=verify)
        payload = cls._payload_class.fromfile(fh, nchan, bps)
        return cls(header, payload, valid, verify)

    @classmethod
    def fromdata(cls, data, header, *args, **kwargs):
        valid = kwargs.pop('valid', True)
        verify = kwargs.pop('verify', True)
        payload = cls._payload_class.fromdata(data, *args, **kwargs)
        if not valid:
            payload.words[...] = cls._fill_pattern
        return cls(header, payload, valid=valid, verify=verify)
    fromdata.__func__.__doc__ = VLBIFrameBase.fromdata.__doc__

    def todata(self, data=None):
        out = super(Mark5BFrame, self).todata(data)
        if not self.valid:
            out[...] = 0.
        return out
