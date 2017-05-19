************
Installation
************


Requirements
============

Baseband requires:

    - `Astropy`_ v1.3.2 or later
    - `NumPy <http://www.numpy.org/>`_ v1.12 or later


Installing baseband
===================

.. Using pip
   ---------

   Baseband currently cannot be built with `pip <http://www.pip-installer.org/en/latest/>`_,
   but eventually...

.. _pipbuildtest:

Testing an install
------------------

As with astropy, you can test if an install has been successful by running the
:meth:`baseband.test` function::

    import baseband
    baseband.test()

Building from source
====================

Obtaining source code
---------------------

Development repository
^^^^^^^^^^^^^^^^^^^^^^

The latest development version of baseband can found on `its GitHub repo <https://github.com/mhvk/baseband>`_;
to clone, use::

    git clone git@github.com:mhvk/baseband.git

Running code without installing
-------------------------------

Baseband can be used without being built or installed, by appending the system
path to include the directory it's located in.  In bash, use

.. code-block:: bash

    export PYTHONPATH="${PYTHONPATH}:BASEBAND_PATH"

and in tcsh, use

.. code-block:: tcsh

    setenv PYTHONPATH ${PYTHONPATH}:BASEBAND_PATH

Alternatively, you can use ``sys.path`` within Python to append the path::

    import sys
    sys.path.append(BASEBAND_PATH)

``BASEBAND_PATH`` is the directory you downloaded or cloned baseband into.


Installing source code
----------------------

To get baseband loaded into your Python working directory, use ``setup.py`` in
the root directory by calling::

    python3 setup.py install

To view baseband-specific options, use ``python3 setup.py --help-commands``.  
For general information on ``setup.py``, see `its documentation
<https://docs.python.org/3.5/install/index.html#install-index>`_ . Some of the 
install options are inherited from Astropy (specifically, from `Astropy-
affiliated package manager <https://github.com/astropy/package-template>`_) and 
are described further in `Astropy's installation documentation 
<https://astropy.readthedocs.io/en/stable/install.html>`_ .

.. _builddocs:

Building documentation
----------------------

.. note::

    As with Astropy, building the documentation is unnecessary unless you
    are writing new documentation or do not have internet access, as baseband's
    documentation is available online at `baseband.readthedocs.io 
    <https://baseband.readthedocs.io>`_ .

Building the documentation requires the Baseband source code (above) and:

    - `Sphinx <http://sphinx.pocoo.org>`_ (and its dependencies) 1.5

There are two ways to build the Astropy documentation. The first is using
``setup.py`` in the root directory::

    python3 setup.py build_docs

The documentation will be built in ``docs/_build/html``, with the main page 
as ``docs/_build/html/index.html``.  To generate LaTeX instead, use::

    python3 setup.py build_docs -b latex

The LaTeX file ``baseband.tex`` will then be created in the 
``docs/_build/latex`` directory.

Using ``setup.py`` builds documentation from source.  To build documentation
using the installed version of baseband (which requires building it, above):
::

    cd docs
    make html

.. _sourcebuildtest:

Testing source code build of baseband
-------------------------------------

The root directory ``setup.py`` can also be used to test if baseband can
successfully be run on your system (regardless of whether or not it's 
installed).  Use::

    python3 setup.py test

These tests require `pytest <http://pytest.org>`_ to be installed.  They are
the same tests as in pipbuildtest_.  Further documentation can be found on
the `Astropy running tests documentation
<https://astropy.readthedocs.io/en/stable/development/testguide.html#running-tests>`_ .

