# Licensed under the GPLv3 - see LICENSE
"""Radio baseband I/O."""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------
__minimum_python_version__ = '3.7'
__minimum_numpy_version__ = '1.17.0'
__minimum_astropy_version__ = '4.0'

if not _ASTROPY_SETUP_:   # noqa
    # For egg_info test builds to pass, put package imports here.
    pass


from .core import file_info, open
