.. _installation:

************
Installation
************

.. _install_reqs:

Requirements
============

Baseband requires:

    - `Astropy`_ v3.0 or later
    - `Numpy <https://www.numpy.org/>`_ v1.10 or later

.. _install_baseband:

Installing Baseband
===================

To install Baseband with `pip <https://pip.pypa.io/>`_,
run::

    pip3 install baseband

.. note::

    To run without pip potentially updating Numpy and Astropy, run, include the
    ``--no-deps`` flag.

Obtaining Source Code
---------------------

The source code and latest development version of Baseband can found on `its
GitHub repo <https://github.com/mhvk/baseband>`_.  You can get your own clone
using::

    git clone git@github.com:mhvk/baseband.git

Of course, it is even better to fork it on GitHub, and then clone your own
repository, so that you can more easily contribute!

Running Code without Installing
-------------------------------

As Baseband is purely Python, it can be used without being built or installed,
by appending the directory it is located in to the ``PYTHON_PATH`` environment
variable.  Alternatively, you can use :obj:`sys.path` within Python to append
the path::

    import sys
    sys.path.append(BASEBAND_PATH)

where ``BASEBAND_PATH`` is the directory you downloaded or cloned Baseband into.

Installing Source Code
----------------------

If you want Baseband to be more broadly available, either to all users on a
system, or within, say, a virtual environment, use :file:`setup.py` in
the root directory by calling::

    python3 setup.py install

For general information on :file:`setup.py`, see `its documentation
<https://docs.python.org/3.5/install/index.html#install-index>`_ . Many of the
:file:`setup.py` options are inherited from Astropy (specifically, from `Astropy
-affiliated package manager <https://github.com/astropy/package-template>`_) and
are described further in `Astropy's installation documentation
<https://astropy.readthedocs.io/en/stable/install.html>`_ .

.. _install_sourcebuildtest:

Testing the Installation
========================

The root directory :file:`setup.py` can also be used to test if Baseband can
successfully be run on your system::

    python3 setup.py test

or, inside of Python::

    import baseband
    baseband.test()

These tests require `pytest <https://pytest.org>`_ to be installed. Further
documentation can be found on the `Astropy running tests documentation
<https://astropy.readthedocs.io/en/stable/development/testguide.html#running-tests>`_
.

.. _install_builddocs:

Building Documentation
======================

.. note::

    As with Astropy, building the documentation is unnecessary unless you
    are writing new documentation or do not have internet access, as Baseband's
    documentation is available online at `baseband.readthedocs.io
    <https://baseband.readthedocs.io>`_.

The Baseband documentation can be built again using :file:`setup.py` from
the root directory::

    python3 setup.py build_docs

This requires to have `Sphinx <https://www.sphinx-doc.org>`_ installed (and its
dependencies).
