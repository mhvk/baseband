# Licensed under the GPLv3 - see LICENSE
"""Mark 4 VLBI data reader.

Code inspired by Walter Brisken's mark5access.  See
https://github.com/demorest/mark5access.

The format itself is described in detail in
http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .base import open
from .header import Mark4Header
from .payload import Mark4Payload
from .frame import Mark4Frame
