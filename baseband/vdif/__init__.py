# Licensed under the GPLv3 - see LICENSE
#
# __      __  _____    _   _____
# \ \    / / | ___ \  | | |   __|
#  \ \  / /  | |  | | | | |  |_
#   \ \/ /   | |  | | | | |   _]
#    \  /    | |__| | | | |  |
#     \/     |_____/  |_| |__|
#
#
"""VLBI Data Interchange Format (VDIF) reader/writer

For the VDIF specification, see http://www.vlbi.org/vdif
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from .base import open
from .header import VDIFHeader
from .payload import VDIFPayload
from .frame import VDIFFrame, VDIFFrameSet
