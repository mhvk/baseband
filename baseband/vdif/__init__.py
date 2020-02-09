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

For the VDIF specification, see https://vlbi.org/vlbi-standards/vdif/
"""
from .base import open  # noqa
from .header import VDIFHeader  # noqa
from .payload import VDIFPayload  # noqa
from .frame import VDIFFrame, VDIFFrameSet  # noqa
