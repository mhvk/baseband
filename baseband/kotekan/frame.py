# Licensed under the GPLv3 - see LICENSE
from ..base.frame import FrameBase
from .header import KotekanHeader
from .payload import KotekanPayload


__all__ = ['KotekanFrame']


class KotekanFrame(FrameBase):
    _header_class = KotekanHeader
    _payload_class = KotekanPayload
