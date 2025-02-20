# Licensed under the GPLv3 - see LICENSE
"""Radio baseband I/O."""

from .io import file_info, open  # noqa

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''

# Define minima for the documentation, but do not bother to explicitly check.
__minimum_python_version__ = '3.10'
__minimum_astropy_version__ = '5.1'
__minimum_numpy_version__ = '1.24'
