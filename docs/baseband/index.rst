.. _baseband_docs:

*************
Documentation
*************

The Baseband package provides:

- File input/output functions for supported radio baseband formats, listed under
  :ref:`specific file formats <specific_file_formats_toc>`.
- Data container classes for supported formats, which can be used for accessing
  and manipulating data samples and for generating baseband formats from
  user-defined data.  The latter functionality can be used for converting
  between file formats.
- Classes for reading from and writing to an ordered sequence of files as if
  it was a single file.

.. _using_baseband_toc:

Using Baseband
==============

.. toctree::
   :maxdepth: 1

   tutorials/getting_started
   tutorials/frame_io

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
   gsb/index

.. _core_utilities_toc:

Core framework and utilities
============================

These sections contain APIs and usage notes for the sequential file opener, and
the API for the set of core utility functions and classes located in
:mod:`~baseband.vlbi_base`.

.. toctree::
   :maxdepth: 1

   helpers/index
   vlbi_base/index

.. _dev_docs_toc:

Developer documentation
=======================

The developer documentation feature tutorials for supporting new formats or
format extensions such as VDIF EDV.

.. toctree::
   :maxdepth: 1

   tutorials/new_edv