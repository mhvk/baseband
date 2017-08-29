******************
User Documentation
******************

Baseband Package
================

Baseband's code is subdivided into its supported file formats.  The following
sections contain tutorials on how to use Baseband for each format.  Also
included is Baseband's API for each format, as well as the sequential
file opener and the set of core utility functions and classes located in
:mod:`~baseband.vlbi_base`.

Specific file formats
---------------------

.. toctree::
   :maxdepth: 1

   vdif/index
   mark5b/index
   mark4/index
   dada/index
   gsb/index

Core framework and utilities
----------------------------

.. toctree::
   :maxdepth: 1

   helpers/index
   vlbi_base/index


Developer documentation
=======================

The developer documentation contain tutorials for supporting new VDIF EDV
formats, and entirely new file formats, within Baseband.

.. toctree::
   :maxdepth: 1

   new_edv_tutorial
