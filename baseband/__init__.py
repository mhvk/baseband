# Licensed under the GPLv3 - see LICENSE
"""Radio baseband I/O."""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    # from example_mod import *
    pass


from .core import file_info, open
