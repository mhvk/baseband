# Licensed under the GPLv3 - see LICENSE
"""Radio baseband I/O."""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

# Define minima for the documentation, but do not bother to explicitly check.
__minimum_python_version__ = '3.10'
__minimum_astropy_version__ = '5.1'
__minimum_numpy_version__ = '1.24'

from .io import file_info, open  # noqa
