from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..vlbi_base import VLBIFrameBase
from .header import Mark5BHeader
from .payload import Mark5BPayload


class Mark5BFrame(VLBIFrameBase):

    _header_class = Mark5BHeader
    _payload_class = Mark5BPayload

    @classmethod
    def fromfile(cls, fh, ref_mjd, nchan, bps=2, verify=True):
        header = cls._header_class.fromfile(fh, ref_mjd, verify=verify)
        payload = cls._payload_class.fromfile(fh, nchan, bps)
        return cls(header, payload, verify)
