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
