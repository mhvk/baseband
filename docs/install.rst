.. _installation:

************
Installation
************

.. _install_reqs:

Requirements
============

Baseband requires:

- Python_ |minimum_python_version| or later
- Astropy_ |minimum_astropy_version| or later
- Numpy_ |minimum_numpy_version| or later
- `entrypoints <https://entrypoints.readthedocs.io/en/latest/>`_

.. _install_baseband:

Installing Baseband
===================

To install Baseband with `pip <https://pip.pypa.io/>`_,
run::

    pip3 install baseband

Here, include a ``--user`` flag if you are installing for yourself
outside of a virtual environment.  You can also add a trailing
``[all]`` to install also the `baseband-tasks`_ package (and all of
its dependencies), which provides analysis tasks and I/O support for
`~baseband_tasks.io.hdf5`.

Obtaining Source Code
---------------------

The source code and latest development version of Baseband can found on `its
GitHub repo <https://github.com/mhvk/baseband>`_.  You can get your own clone
using::

    git clone git@github.com:mhvk/baseband.git

Of course, it is even better to fork it on GitHub, and then clone your own
repository, so that you can more easily contribute!  You can install the
cloned repository with::

  pip3 install .

Here, apart from the ``--user`` option, you may want to add the ``--editable``
option to just link to the source repository, which means that any edit will
be seen.

Running Code without Installing
-------------------------------

As Baseband is purely Python, it can be used without being built or installed,
by appending the directory it is located in to the ``PYTHON_PATH`` environment
variable.  Alternatively, you can use :obj:`sys.path` within Python to append
the path::

    import sys
    sys.path.append(BASEBAND_PATH)

where ``BASEBAND_PATH`` is the directory you downloaded or cloned Baseband into.
However, for its plugin modules to be discoverable, you may still want to run::

    python3 setup.py egg_info


.. _install_sourcebuildtest:

Testing the Installation
========================

To test that the code works on your system, you need
`pytest <http://pytest.org>`_ and
`pytest-astropy <https://github.com/astropy/pytest-astropy>`_
to be installed;
this is most easily done by first installing the code together
with its test dependencies::

    pip install -e .[test]

Then, inside the root directory, simply run

    pytest

or, inside of Python::

    import baseband
    baseband.test()

For further details, see the `Astropy Running Tests pages
<https://astropy.readthedocs.io/en/latest/development/testguide.html#running-tests>`_.

.. _install_builddocs:

Building Documentation
======================

.. note::

    As with Astropy, building the documentation is unnecessary unless you
    are writing new documentation or do not have internet access, as
    Baseband's documentation is available online at
    `baseband.readthedocs.io <https://baseband.readthedocs.io>`_.

To build the Baseband documentation, you need
`Sphinx <http://sphinx.pocoo.org>`_ and
`sphinx-astropy <https://github.com/astropy/sphinx-astropy>`_
to be installed;
this is most easily done by first installing the code together
with its documentations dependencies::

    pip install -e .[docs]

Then, go to the ``docs`` directory and run

    make html

For further details, see the `Astropy Building Documentation pages
<http://docs.astropy.org/en/latest/install.html#builddocs>`_.
