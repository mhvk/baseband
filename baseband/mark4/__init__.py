"""Mark 4 VLBI data reader.  Code inspired by Walter Brisken's mark5access.
See https://github.com/demorest/mark5access.

The format itself is described in detail in
http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .header import Mark4Header
from .data import Mark4Data
