.. _mark4:

.. include:: ../tutorials/glossary_substitutions.rst

******
MARK 4
******

The Mark 4 format is the output format of the MIT Haystack Observatory's Mark 4
VLBI magnetic tape-based data acquisition system, and one output format of its
successor, the Mark 5A hard drive-based system. The format's specification is
in the Mark IIIA/IV/VLBA `design specifications`_.

Baseband currently only supports files that have been parity-stripped and
corrected for barrel roll and data modulation.

.. _design specifications: https://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf

.. _mark4_file_structure:

File Structure
==============

Mark 4 files contain up to 64 concurrent data "tracks".  Tracks are divided
into 22500-bit "tape frames", each of which consists of a 160-bit
:term:`header` followed by a 19840-bit :term:`payload`.  The header includes a
timestamp (accurate to 1.25 ms), track ID, sideband, and fan-out/in factor
(see below); the details of these can be found in 2.1.1 - 2.1.3 in the
`design specifications`_.  The payload consists of a 1-bit :term:`stream`.
When recording 2-bit |elementary samples|, the data is split into two tracks,
with one carrying the sign bit, and the other the magnitude bit.

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
usage can be found under the :ref:`Using Baseband <using_baseband>` section.
For situations in which one is unsure of a file's format, Baseband features the
general `baseband.open` and `baseband.file_info` functions, which are also
discussed in :ref:`Using Baseband <using_baseband>`.  The examples below use
the small sample file ``baseband/data/sample.m4``, and the `numpy`,
`astropy.units`, `astropy.time.Time`, and `baseband.mark4` modules::

    >>> import numpy as np
    >>> import astropy.units as u
    >>> from astropy.time import Time
    >>> from baseband import mark4
    >>> from baseband.data import SAMPLE_MARK4

Opening a Mark 4 file with `~baseband.mark4.open` in binary mode provides
a normal file reader but extended with methods to read a
`~baseband.mark4.Mark4Frame`.  Mark 4 files generally **do not start (or end)
at a frame boundary**, so in binary mode one has to find the first
header using `~baseband.mark4.base.Mark4FileReader.find_header` (which will
also determine the number of Mark 4 tracks, if not given explicitly). Since
Mark 4 files do not store the full time information, one must pass either the
the decade the data was taken, or an equivalent reference `~astropy.time.Time`
object::

    >>> fb = mark4.open(SAMPLE_MARK4, 'rb', decade=2010)
    >>> fb.find_header()  # Locate first header and determine ntrack.
    <Mark4Header bcd_headstack1: [0x3344]*64,
                 bcd_headstack2: [0x1122]*64,
                 headstack_id: [0, ..., 1],
                 bcd_track_id: [0x2, ..., 0x33],
                 fan_out: [0, ..., 3],
                 magnitude_bit: [False, ..., True],
                 lsb_output: [True]*64,
                 converter_id: [0, ..., 7],
                 time_sync_error: [False]*64,
                 internal_clock_error: [False]*64,
                 processor_time_out_error: [False]*64,
                 communication_error: [False]*64,
                 _1_11_1: [False]*64,
                 _1_10_1: [False]*64,
                 track_roll_enabled: [False]*64,
                 sequence_suspended: [False]*64,
                 system_id: [108]*64,
                 _1_0_1_sync: [False]*64,
                 sync_pattern: [0xffffffff]*64,
                 bcd_unit_year: [0x4]*64,
                 bcd_day: [0x167]*64,
                 bcd_hour: [0x7]*64,
                 bcd_minute: [0x38]*64,
                 bcd_second: [0x12]*64,
                 bcd_fraction: [0x475]*64,
                 crc: [0xea6, ..., 0x212]>
    >>> fb.ntrack
    64
    >>> fb.tell()
    2696
    >>> frame = fb.read_frame()
    >>> frame.shape
    (80000, 8)
    >>> frame.header.time
    <Time object: scale='utc' format='yday' value=2014:167:07:38:12.47500>
    >>> fb.close()

Opening in stream mode automatically finds the first frame, and wraps the
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

    >>> fw = mark4.open('sample_mark4_segment.m4', 'ws', header0=frame.header,
    ...                 sample_rate=32*u.MHz, decade=2010)
    >>> fw.write(frame.data)
    >>> fw.close()
    >>> fh = mark4.open('sample_mark4_segment.m4', 'rs',
    ...                 sample_rate=32.*u.MHz, decade=2010)
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
.. automodapi:: baseband.mark4.file_info
.. automodapi:: baseband.mark4.base
