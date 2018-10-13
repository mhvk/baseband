2.0 (unreleased)
================

- VDIF and Mark 5B readers and writers now support 1 bit per sample.
  [#277, #278]

Other Changes and Additions
---------------------------

- Added release procedure to the documentation.  [#268]

1.2.1 (unreleased)
==================

Bug Fixes
---------

- VDIF reader will now properly ignore corrupt last frames. [#273]

- Mark5B reader more robust against headers not being parsed correctly
  in ``Mark5BFileReader.find_header``. [#275]

- All stream readers now have a proper ``dtype`` attribute, not a
  corresponding ``np.float32`` or ``np.complex64``. [#280]

1.2 (2018-07-27)
================

New Features
------------

- Expanded support for acccessing sequences of files to VLBI format
  openers and `baseband.open`.  Enabled `baseband.guppi.open` to open file
  sequences using string templates like with `baseband.dada.open`. [#254]

- Created `baseband.helpers.sequentialfile.FileNameSequencer`, a
  general-purpose filename sequencer that can be passed to any format opener.
  [#253]

Other Changes and Additions
---------------------------

- Moved the Getting Started section to :ref:`"Using Baseband"
  <using_baseband>`, and created a new quickstart tutorial under :ref:`Getting
  Started <getting_started>` to better assist new users.  [#260]

1.1.1 (2018-07-24)
==================

Bug Fixes
---------

- Ensure ``gsb`` times can be decoded with astropy-dev (which is to become
  astropy 3.1). [#249]

- Fixed rounding error when encoding 4-bit data using
  `baseband.vlbi_base.encoding.encode_4bit_base`. [#250]

- Added GUPPI/PUPPI to the list of file formats used by `baseband.open` and
  `baseband.file_info`.  [#251]

1.1 (2018-06-06)
================

New Features
------------

- Added a new `baseband.file_info` function, which can be used to inspect
  data files. [#200]

- Added a general file opener, `baseband.open` which for a set of formats
  will check whether the file is of that format, and then load it using the
  corresponding module. [#198]

- Allow users to pass a ``verify`` keyword to file openers reading streams.
  [#233]

- Added support for the GUPPI format. [#212]

- Enabled `baseband.dada.open` to read streams where the last frame has an
  incomplete payload. [#228]

API Changes
-----------

- In analogy with Mark 5B, VDIF header time getting and setting now requires
  a frame rate rather than a sample rate. [#217, #218]

- DADA and GUPPI now support passing either a ``start_time`` or ``offset``
  (in addition to ``time``) to set the start time in the header. [#240]

Bug Fixes
---------

Other Changes and Additions
---------------------------

- The `baseband.data` module with sample data files now has an explicit entry
  in the documentation. [#198]

- Increased speed of VLBI stream reading by changing the way header sync
  patterns are stored, and removing redundant verification steps.  VDIF
  sequential decode is now 5 - 10% faster (depending on the number of
  threads). [#241]

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
