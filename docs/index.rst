.. _baseband_docs:

********
Baseband
********

Welcome to the Baseband documentation! Baseband is a package for reading and
writing VLBI and other radio baseband files, with the aim of simplifying and
streamlining data conversion and standardization.  It provides:

- File input/output objects for supported radio baseband formats, enabling
  selective decoding of data into `Numpy arrays <numpy.ndarray>`, and encoding
  user-defined arrays into baseband formats. Supported formats are listed under
  :ref:`specific file formats <specific_file_formats_toc>`.
- Helper objects for reading from and writing to an ordered sequence of files
  as if it was a single file.

.. _overview_toc:

Overview
========

.. toctree::
   :maxdepth: 1

   install
   tutorials/getting_started
   tutorials/glossary

.. _specific_file_formats_toc:

Specific file formats
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

Core framework and utilities
============================

These sections contain APIs and usage notes for the sequential file opener,
the API for the set of core utility functions and classes located in
:mod:`~baseband.vlbi_base`, and sample data that come with baseband (mostly
used for testing).

.. toctree::
   :maxdepth: 1

   helpers/index
   vlbi_base/index
   data/index

.. _dev_docs_toc:

Developer documentation
=======================

The developer documentation feature tutorials for supporting new formats or
format extensions such as VDIF EDV.

.. toctree::
   :maxdepth: 1

   tutorials/new_edv

.. _project_details_toc:

Project details
===============

.. image:: https://travis-ci.org/mhvk/baseband.svg?branch=master

.. image:: https://coveralls.io/repos/github/mhvk/baseband/badge.svg
   :target: https://coveralls.io/github/mhvk/baseband

.. image:: https://readthedocs.org/projects/baseband/badge/?version=latest
   :target: http://baseband.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. toctree::
   :maxdepth: 1

   authors_for_sphinx
   changelog
   license

Reference/API
=============
.. automodapi:: baseband
