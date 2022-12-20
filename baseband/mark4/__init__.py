# Licensed under the GPLv3 - see LICENSE
"""Mark 4 VLBI data reader.

Code inspired by Walter Brisken's mark5access.  See
https://github.com/difx/mark5access.

The format itself is described in detail in
https://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
"""
from .base import open, info  # noqa
from .header import Mark4Header  # noqa
from .payload import Mark4Payload  # noqa
from .frame import Mark4Frame  # noqa
