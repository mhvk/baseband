# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure from the pythong interpreter (i.e., in baseband.test()).
import os

try:
    from pytest_astropy_header.display import (PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)
except ImportError:
    pass
else:
    def pytest_configure(config):

        config.option.astropy_header = True

        # Customize the following lines to add/remove entries from the list of
        # packages for which version numbers are displayed when running the
        # tests inside the python interpreter.
        PYTEST_HEADER_MODULES.clear()
        PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
        PYTEST_HEADER_MODULES['Numpy'] = 'numpy'

        try:
            from .version import version
        except ImportError:  # Can happen in source checkout.
            version = 'from source'

        packagename = os.path.basename(os.path.dirname(__file__))
        TESTED_VERSIONS[packagename] = version

# Uncomment the last two lines in this block to treat all DeprecationWarnings as
# exceptions. For Astropy v2.0 or later, there are 2 additional keywords,
# as follow (although default should work for most cases).
# To ignore some packages that produce deprecation warnings on import
# (in addition to 'compiler', 'scipy', 'pygments', 'ipykernel', and
# 'setuptools'), add:
#     modules_to_ignore_on_import=['module_1', 'module_2']
# To ignore some specific deprecation warning messages for Python version
# MAJOR.MINOR or later, add:
#     warnings_to_ignore_by_pyver={(MAJOR, MINOR): ['Message to ignore']}
# from astropy.tests.helper import enable_deprecations_as_exceptions  # noqa
# enable_deprecations_as_exceptions()
