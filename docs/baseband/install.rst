************
Installation
************


Requirements
============

Baseband requires:

    - `Astropy`_ v1.0.4 or later
    - `Numpy <http://www.numpy.org/>`_ v1.9 or later

.. _installation:

Installing Baseband
===================

.. Using pip
   ---------

   Baseband currently cannot be built with `pip <http://www.pip-installer.org/en/latest/>`_,
   but eventually...

Currently, baseband can only be installed by getting its source code,
and either running it directly or installing it.

Obtaining source code
---------------------

The source code and latest development version of baseband can found on `its
GitHub repo <https://github.com/mhvk/baseband>`_.  You can get your own clone
using::

    git clone git@github.com:mhvk/baseband.git

Of course, it is even better to fork it on GitHub, and then clone your own
repository, so that you can more easily contribute!

Running code without installing
-------------------------------

As baseband is purely Python, it can be used without being built or installed,
by appending the directory it is located in to the ``PYTHON_PATH`` environment
variable.  Alternatively, you can use :obj:`sys.path` within Python to append 
the path::

    import sys
    sys.path.append(BASEBAND_PATH)

where ``BASEBAND_PATH`` is the directory you downloaded or cloned baseband into.

Installing source code
----------------------

If you want baseband to be more broadly available, either to all users on a
system, or within, say, a virtual environment, use :file:`setup.py` in
the root directory by calling::

    python3 setup.py install

For general information on :file:`setup.py`, see `its documentation
<https://docs.python.org/3.5/install/index.html#install-index>`_ . Many of the
:file:`setup.py` options are inherited from Astropy (specifically, from `Astropy
-affiliated package manager <https://github.com/astropy/package-template>`_) and
are described further in `Astropy's installation documentation
<https://astropy.readthedocs.io/en/stable/install.html>`_ .

.. _sourcebuildtest:

Testing the installation
========================

The root directory :file:`setup.py` can also be used to test if baseband can
successfully be run on your system::

    python3 setup.py test

or, inside of Python::

    import baseband
    baseband.test()

These tests require `pytest <http://pytest.org>`_ to be installed. Further
documentation can be found on the `Astropy running tests documentation
<https://astropy.readthedocs.io/en/stable/development/testguide.html#running-tests>`_
.

.. _builddocs:

Building documentation
======================

.. note::

    As with Astropy, building the documentation is unnecessary unless you
    are writing new documentation or do not have internet access, as baseband's
    documentation is available online at `baseband.readthedocs.io 
    <https://baseband.readthedocs.io>`_.

The baseband documentation can be built again using :file:`setup.py` from 
the root directory::

    python3 setup.py build_docs

This requires to have `Sphinx <http://sphinx.pocoo.org>`_ installed (and its
dependencies; version 1.5 recommended).
