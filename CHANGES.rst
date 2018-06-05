1.1 (unreleased)
================

New Features
------------

- Added a new `baseband.file_info` function, which can be used to inspect
  data files. [#200]

- Added a general file opener, `baseband.open` which for a set of formats
  will check whether the file is of that format, and then load it using the
  corresponding module. [#198]

- Added support for the GUPPI format. [#212]

- Enabled `baseband.dada.open` to read streams where the last frame has an
  incomplete payload. [#228]

API Changes
-----------

- In analogy with Mark 5B, VDIF header time getting and setting now requires
  a frame rate rather than a sample rate. [#217]

Bug Fixes
---------

Other Changes and Additions
---------------------------

- The `baseband.data` module with sample data files now has an explicit entry
  in the documentation. [#198]

1.0.1 (2018-06-04)
==================

Bug Fixes
---------

- Fixed a bug in `baseband.dada.open` where passing a ``squeeze`` setting is
  ignored when also passing header keywords in 'ws' mode. [#211]

- Raise an exception rather than return incorrect times for Mark 5B files
  in which the fractional seconds are not set. [#216]

Other Changes and Additions
---------------------------

- Fixed broken links and typos in the documentation. [#211]


1.0.0 (2018-04-09)
==================

- Initial release.
