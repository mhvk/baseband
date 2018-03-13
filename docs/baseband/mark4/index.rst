.. _mark4:

.. include:: ../tutorials/glossary_substitutions.rst

******
MARK 4
******

The Mark 4 format is the output format of the MIT Haystack Observatory's Mark 4
VLBI magnetic tape-based data acquisition system, and one output format of its
successor, the Mark 5A hard drive-based system. The format's specification is
in the `Mark IIIA/IV/VLBA documentation <m4spec>`_.

Baseband currently only supports files that have been parity-stripped and
corrected for barrel roll and data modulation.

.. _m4spec: http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf

.. _mark4_file_structure:

File Structure
==============

Mark 4 files contain up to 64 concurrent data "tracks".  Tracks are divided
into 22500-bit "tape frames", each of which consists of a 160-bit
:term:`header` followed by a 19840-bit :term:`payload`.  The header includes a
timestamp (accurate to 1.25 ms), track ID, sideband, and fan-out/in factor
(see below); the details of these can be found in 2.1.1 - 2.1.3 in the
`specification document <m4spec>`_.  The payload consists of a 1-bit
:term:`stream`.  When recording 2-bit |elementary samples|, the data is split
into two tracks, with one carrying the sign bit, and the other the magnitude
bit.

The header takes the place of the first 160 bits of payload data, so that the
first sample occurs ``fanout * 160`` sample times after the header time.  This
means that a Mark 4 stream is not contiguous in time.  The length of
one frame ranges from 1.25 ms to 160 ms in octave steps (which ensures an
integer number of frames falls within 1 minute), setting the maximum sample
rate per track to 18 megabits/track/s.

Data from a single :term:`channel` may be distributed to multiple tracks -
"fan-out" - or multiple channels fed to one track - "fan-in".  Fan-out
is used when sampling at rates higher than 18 megabits/track/s.  Baseband
currently only supports tracks using fan-out ("longitudinal data format").

Baseband reconstructs the tracks into channels (reconstituting 2-bit data from
two tracks into a single channel if necessary) and combines tape
frame headers into a single :term:`data frame` header.

.. _mark4_usage:

Usage
=====

This section covers reading and writing Mark 4 files with Baseband; general
usage can be found under the :ref:`Using Baseband <using_baseband_toc>` section.
The examples below use the small sample file ``baseband/data/sample.m4``, and
the `numpy`, `astropy.units`, `astropy.time.Time`, and `baseband.mark4`
modules::

    >>> import numpy as np
    >>> import astropy.units as u
    >>> from astropy.time import Time
    >>> from baseband import mark4
    >>> from baseband.data import SAMPLE_MARK4

Opening a Mark 4 file with `~baseband.mark4.open` in binary mode provides
a normal file reader but extended with methods to read a
`~baseband.mark4.Mark4Frame`.  Mark 4 files generally **do not start (or end)
at a frame boundary**, so in binary mode one has to seek the
first frame using `~baseband.mark4.base.Mark4FileReader.find_frame`.  To use
`~baseband.mark4.base.Mark4FileReader.find_frame` and 
`~baseband.mark4.base.Mark4FileReader.read_frame`, one must pass both the
number of tracks, and either the decade the data was taken, or an equivalent
reference `~astropy.time.Time` object, since those numbers cannot be inferred
from the data themselves::

    >>> fb = mark4.open(SAMPLE_MARK4, 'rb')
    >>> fb.find_frame(ntrack=64)    # Find first frame.
    2696
    >>> frame = fb.read_frame(ntrack=64, decade=2010)
    >>> frame.shape
    (80000, 8)

If one does not know the number of tracks, one can attempt to determine
this using `~baseband.mark4.base.Mark4FileReader.determine_ntracks`
(which will leave the file pointer at the start of a frame as well)::

    >>> fb.seek(0)
    0
    >>> ntrack = fb.determine_ntrack()    # Also finds first frame.
    >>> ntrack
    64
    >>> frame2 = fb.read_frame(ntrack=ntrack, decade=2010)
    >>> frame2 == frame
    True
    >>> fb.close()

Opening in stream mode automatically seeks for the first frame (determing
the number of tracks if not given explicitly), and wraps the
low-level routines such that reading and writing is in units of samples.  It
also provides access to header information.  Here we pass a reference
`~astropy.time.Time` object within 4 years of the observation start time to
``ref_time``, rather than a ``decade``::

    >>> fh = mark4.open(SAMPLE_MARK4, 'rs', ref_time=Time('2013:100:23:00:00'))
    >>> fh
    <Mark4StreamReader name=... offset=0
        sample_rate=32.0 MHz, samples_per_frame=80000,
        sample_shape=SampleShape(nchan=8), bps=2,
        start_time=2014-06-16T07:38:12.47500>
    >>> d = fh.read(6400)
    >>> d.shape
    (6400, 8)
    >>> d[635:645, 0].astype(int)  # first channel
    array([ 0,  0,  0,  0,  0, -1,  1,  3,  1, -1])
    >>> fh.close()

As mentioned in the :ref:`mark4_file_structure` section, because the header
takes the place of the first 160 samples of each track, the first payload
sample occurs ``fanout * 160`` sample times after the header time.  The stream
reader includes these overwritten samples as invalid data (zeros, by default)::

    >>> np.array_equal(d[:640], np.zeros((640,) + d.shape[1:]))
    True

When writing to file, we need to pass in the sample rate in addition
to ``decade``.  The number of tracks can be inferred from the header::

    >>> fw = mark4.open('sample_mark4_segment.m4', 'ws', header=frame.header,
    ...                 decade=2010, sample_rate=32*u.MHz)
    >>> fw.write(frame.data)
    >>> fw.close()
    >>> fh = mark4.open('sample_mark4_segment.m4', 'rs', decade=2010,
    ...                 sample_rate=32.*u.MHz)
    >>> np.all(fh.read(80000) == frame.data)
    True
    >>> fh.close()

Note that above we had to pass in the sample rate even when opening
the file for reading; this is because there is only a single frame in
the file, and hence the sample rate cannot be inferred automatically.

.. _mark4_api:

Reference/API
=============

.. automodapi:: baseband.mark4
.. automodapi:: baseband.mark4.header
   :include-all-objects:
.. automodapi:: baseband.mark4.payload
.. automodapi:: baseband.mark4.frame
.. automodapi:: baseband.mark4.base
