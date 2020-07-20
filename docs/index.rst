.. _baseband_docs:

********
Baseband
********

Welcome to the Baseband documentation!  Baseband is a package
`affiliated <https://www.astropy.org/affiliated/index.html>`_ with the
Astropy_ project for reading and
writing VLBI and other radio baseband files, with the aim of simplifying and
streamlining data conversion and standardization.  It provides:

- File input/output objects for supported radio baseband formats, enabling
  selective decoding of data into `Numpy arrays <numpy.ndarray>`, and encoding
  user-defined arrays into baseband formats. Supported formats are listed under
  :ref:`specific file formats <specific_file_formats_toc>`.
- The ability to read from and write to an ordered sequence of files as if it
  was a single file.

It can be extended with the more experimental baseband-tasks_ package,
which provides tasks to, e.g., `~baseband_tasks.channelize.Channelize`
or `~baseband_tasks.dispersion.Dedisperse` sample streams.

If you used this package in your research, please cite it via DOI
`10.5281/zenodo.1214268 <https://doi.org/10.5281/zenodo.1214268>`_.

.. _overview_toc:

Overview
========

.. toctree::
   :maxdepth: 2

   install
   tutorials/getting_started
   tutorials/using_baseband
   tutorials/glossary

.. _specific_file_formats_toc:

Specific File Formats
=====================

Baseband's code is subdivided into its supported file formats, and the
following sections contain format specifications, usage notes,
troubleshooting help and APIs for each.

.. toctree::
   :maxdepth: 1

   vdif/index
   mark5b/index
   mark4/index
   dada/index
   guppi/index
   gsb/index

.. _core_utilities_toc:

Core Framework and Utilities
============================

These sections contain APIs and usage notes for the sequential file opener,
the API for the set of core utility functions and classes located in
:mod:`~baseband.base`, and sample data that come with baseband (mostly
used for testing).

.. toctree::
   :maxdepth: 1

   helpers/index
   base/index
   data/index

.. _dev_docs_toc:

Developer Documentation
=======================

The developer documentation features tutorials for supporting format
extensions such as VDIF EDV or a completely new format, possibly making it
available as a `baseband.io` plugin.  It also contains instructions for
publishing new code releases.

.. toctree::
   :maxdepth: 1

   tutorials/new_edv
   tutorials/new_format
   tutorials/release_procedure

.. _project_details_toc:

Project Details
===============

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1214268.svg
   :target: https://doi.org/10.5281/zenodo.1214268
   :alt: DOI 10.5281/zenodo.1214268

.. image:: https://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: https://www.astropy.org/
    :alt: Powered by Astropy

.. image:: https://travis-ci.org/mhvk/baseband.svg?branch=master
   :target: https://travis-ci.org/mhvk/baseband
   :alt: Test Status

.. image:: https://codecov.io/gh/mhvk/baseband/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/mhvk/baseband
   :alt: Coverage Level

.. image:: https://readthedocs.org/projects/baseband/badge/?version=latest
   :target: https://baseband.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. toctree::
   :maxdepth: 1

   authors_for_sphinx
   changelog
   license

Reference/API
=============
.. automodapi:: baseband
.. automodapi:: baseband.io
.. automodapi:: baseband.tasks
