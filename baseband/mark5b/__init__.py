# Licensed under the GPLv3 - see LICENSE
"""Mark5B VLBI data reader.

Code inspired by Walter Brisken's mark5access.  See
https://github.com/demorest/mark5access.

Also, for the Mark5B design, see
http://www.haystack.mit.edu/tech/vlbi/mark5/mark5_memos/019.pdf
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from .base import open
from .header import Mark5BHeader
from .payload import Mark5BPayload
from .frame import Mark5BFrame
