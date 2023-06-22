# Licensed under the GPLv3 - see LICENSE
"""Radio baseband I/O."""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

# Define minima for the documentation, but do not bother to explicitly check.
__minimum_python_version__ = '3.8'
__minimum_astropy_version__ = '5.0'
__minimum_numpy_version__ = '1.18'

from .io import file_info, open  # noqa
