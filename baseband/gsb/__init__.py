# Licensed under the GPLv3 - see LICENSE
"""GMRT Software Backend (GSB) data reader.

See http://gmrt.ncra.tifr.res.in/gmrt_hpage/sub_system/gmrt_gsb/index.htm
"""
from .base import open  # noqa
from .header import GSBHeader  # noqa
from .payload import GSBPayload  # noqa
from .frame import GSBFrame  # noqa
