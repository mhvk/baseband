.. _guppi:

.. include:: ../tutorials/glossary_substitutions.rst

*****
GUPPI
*****

The GUPPI format is the output of the `Green Bank Ultimate Pulsar Processing
Instrument <https://safe.nrao.edu/wiki/bin/view/CICADA/NGNPP>`_ and any clones
operating at other telescopes, such as `PUPPI at the Arecibo Observatory
<https://www.naic.edu/puppi-observing/>`_.  Baseband specifically supports
GUPPI data **taken in baseband mode**, and is based off of `DSPSR's
implementation <https://github.com/demorest/dspsr>`_.  While general format
specifications can be found on `Paul Demorest's site
<https://www.cv.nrao.edu/~pdemores/GUPPI_Raw_Data_Format>`_, some of the
header information could be invalid or not applicable, particularly with older
files.

Baseband currently only supports 8-bit |elementary samples|.

.. _guppi_file_structure:

File Structure
==============

Each GUPPI file contains multiple (typically 128) |frames|, with each frame
consisting of an ASCII :term:`header` composed of 80-character entries,
followed by a binary :term:`payload` (or "block").  The header's length is
variable, but always ends with "END" followed by 77 spaces.

How samples are stored in the payload depends on whether or not it is
**channels-first**.  A channels-first payload stores each channel's
:term:`stream` in a contiguous data block, while a non-channels-first one
groups the |components| of a :term:`complete sample` together (like with other
formats).  In either case, for each channel polarization samples from the same
point in time are stored adjacent to one another.  At the end of each channel's
data is a section of **overlap samples** identical to the first samples in
the next payload.  Baseband retains these redundant samples when reading
individual GUPPI frames, but removes them when reading files as a stream.

.. _guppi_usage:

Usage
=====

This section covers reading and writing GUPPI files with Baseband; general
usage is covered in the :ref:`Using Baseband <using_baseband>` section.  For
situations in which one is unsure of a file's format, Baseband features the
general `baseband.open` and `baseband.file_info` functions, which are also
discussed in :ref:`Using Baseband <using_baseband>`.  The examples below use
the sample PUPPI file ``baseband/data/sample_puppi.raw``, and the the
`astropy.units` and `baseband.guppi` modules::

    >>> from baseband import guppi
    >>> import astropy.units as u
    >>> from baseband.data import SAMPLE_PUPPI

Single files can be opened with `~baseband.guppi.open` in binary mode, which
provides a normal file reader, but extended with methods to read a
`~baseband.guppi.GUPPIFrame`::

    >>> fb = guppi.open(SAMPLE_PUPPI, 'rb')
    >>> frame = fb.read_frame()
    >>> frame.shape
    (1024, 2, 4)
    >>> frame[:3, 0, 1]    # doctest: +FLOAT_CMP
    array([-32.-10.j, -15.-14.j,   9.-13.j], dtype=complex64)
    >>> fb.close()

Since the files can be quite large, the payload is mapped (with
`numpy.memmap`), so that if one accesses part of the data, only the
corresponding parts of the encoded payload are loaded into memory (since the
sample file is encoded using 8 bits, the above example thus loads 6 bytes into
memory).

Opening in stream mode wraps the low-level routines such that reading and
writing is in units of samples, and provides access to header information::

    >>> fh = guppi.open(SAMPLE_PUPPI, 'rs')
    >>> fh
    <GUPPIStreamReader name=... offset=0
        sample_rate=250.0 Hz, samples_per_frame=960,
        sample_shape=SampleShape(npol=2, nchan=4), bps=8,
        start_time=2018-01-14T14:11:33.000>
    >>> d = fh.read()
    >>> d.shape
    (3840, 2, 4)
    >>> d[:3, 0, 1]    # doctest: +FLOAT_CMP
    array([-32.-10.j, -15.-14.j,   9.-13.j], dtype=complex64)
    >>> fh.close()

Note that ``fh.samples_per_frame`` represents the number of samples per frame
**excluding overlap samples**, since the stream reader works on a linearly
increasing sequence of samples.  Frames themselves have access to the overlap,
and ``fh.header0.samples_per_frame`` returns the number of samples per frame
including overlap.

To set up a file for writing as a stream is possible as well.  Overlap must be
zero when writing (so we set ``samples_per_frame`` to its stream reader value
from above)::

    >>> from astropy.time import Time
    >>> fw = guppi.open('puppi_test.{file_nr:04d}.raw', 'ws',
    ...                 frames_per_file=2, sample_rate=250*u.Hz,
    ...                 samples_per_frame=960, pktsize=1024,
    ...                 time=Time(58132.59135416667, format='mjd'),
    ...                 npol=2, nchan=4)
    >>> fw.write(d)
    >>> fw.close()
    >>> fr = guppi.open('puppi_test.{file_nr:04d}.raw', 'rs')
    >>> d2 = fr.read()
    >>> (d == d2).all()
    True
    >>> fr.close()

Here we show how to write a sequence of files by passing a string template
to `~baseband.guppi.open`, which prompts it to create and use a filename
sequencer generated with `~baseband.guppi.base.GUPPIFileNameSequencer`.  One
may also pass a time-ordered list or tuple of filenames to
`~baseband.guppi.open`.  Unlike when writing DADA files, which have one frame
per file, we specify the number of frames in one file using``frames_per_file``.
Note that typically one does not have to pass ``PKTSIZE``, the UDP data packet
size (set by the observing mode), but the sample file has small enough frames
that the default of 8192 bytes is too large.  Baseband only uses ``PKTSIZE`` to
double-check the sample offset of the frame, so ``PKTSIZE`` must be set to a
value such that each payload, excluding overlap samples, contains an integer
number of packets.  (See API links for further details on how to read and
write file sequences.)

.. _guppi_api:

Reference/API
=============

.. automodapi:: baseband.guppi
.. automodapi:: baseband.guppi.header
.. automodapi:: baseband.guppi.payload
.. automodapi:: baseband.guppi.frame
.. automodapi:: baseband.guppi.file_info
.. automodapi:: baseband.guppi.base
