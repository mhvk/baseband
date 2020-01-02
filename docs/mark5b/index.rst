.. _mark5b:

.. include:: ../tutorials/glossary_substitutions.rst

*******
MARK 5B
*******

The Mark 5B format is the output format of the Mark 5B disk-based VLBI data
system.  It is described in its `design specifications`_.

.. _design specifications: https://www.haystack.mit.edu/tech/vlbi/mark5/mark5_memos/019.pdf

.. _mark5b_file_structure:

File Structure
==============

Each :term:`data frame` consists of a :term:`header` consisting of four 32-bit
words (16 bytes) followed by a :term:`payload` of 2500 32-bit words (10000
bytes).  The header contains a sync word, frame number, and timestamp
(accurate to 1 ms), as well as user-specified data; see Sec. 1 of the
`design specifications`_ for details.  The payload supports :math:`2^n` bit
streams, for :math:`0 \leq n \leq 5`, and the first sample of each stream
corresponds precisely to the header time.  |Elementary samples| may be 1 or 2
bits in size, with the latter being stored in two successive bit streams.  The
number of |channels| is equal to the number of bit-streams divided by the
number of bits per elementary sample (Baseband currently only supports files
where all bit-streams are active).  Files begin at a header (unlike for Mark
4), and an integer number of frames fit within 1 second.

The Mark 5B system also outputs files with the active bit-stream mask, number
of frames per second, and observational metadata (Sec. 1.3 of the `design
specifications`_).  Baseband does not yet use these files, and instead
requires the user specify, for example, the :term:`sample rate`.

.. _mark5b_usage:

Usage
=====

This section covers reading and writing Mark 5B files with Baseband; general
usage can be found under the :ref:`Using Baseband <using_baseband>` section.
For situations in which one is unsure of a file's format, Baseband features the
general `baseband.open` and `baseband.file_info` functions, which are also
discussed in :ref:`Using Baseband <using_baseband>`.  The examples below use
the small sample file ``baseband/data/sample.m5b``, and the `numpy`,
`astropy.units`, `astropy.time.Time`, and `baseband.mark5b` modules::

    >>> import numpy as np
    >>> import astropy.units as u
    >>> from astropy.time import Time
    >>> from baseband import mark5b
    >>> from baseband.data import SAMPLE_MARK5B

Opening a Mark 5B file with `~baseband.mark5b.open` in binary mode provides a
normal file reader extended with methods to read a
`~baseband.mark5b.Mark5BFrame`.  The number of channels, kiloday (thousands of
MJD) and number of bits per sample must all be passed when using
`~baseband.mark5b.base.Mark5BFileReader.read_frame`::

    >>> fb = mark5b.open(SAMPLE_MARK5B, 'rb', kday=56000, nchan=8)
    >>> frame = fb.read_frame()
    >>> frame.shape
    (5000, 8)
    >>> fb.close()

Our sample file has 2-bit :term:`component` samples, which is also the default
for `~baseband.mark5b.base.Mark5BFileReader.read_frame`, so it does not need to
be passed.  Also, we may pass a reference `~astropy.time.Time` object within
500 days of the observation start time to ``ref_time``, rather than ``kday``.

Opening as a stream wraps the low-level routines such that reading and writing
is in units of samples.  It also provides access to header information.  Here,
we also must provide ``nchan``, ``sample_rate``, and ``ref_time`` or ``kday``::

    >>> fh = mark5b.open(SAMPLE_MARK5B, 'rs', sample_rate=32*u.MHz, nchan=8,
    ...                  ref_time=Time('2014-06-13 12:00:00'))
    >>> fh
    <Mark5BStreamReader name=... offset=0
        sample_rate=32.0 MHz, samples_per_frame=5000,
        sample_shape=SampleShape(nchan=8), bps=2,
        start_time=2014-06-13T05:30:01.000000000>
    >>> header0 = fh.header0    # To be used for writing, below.
    >>> d = fh.read(10000)
    >>> d.shape
    (10000, 8)
    >>> d[0, :3]    # doctest: +FLOAT_CMP
    array([-3.316505, -1.      ,  1.      ], dtype=float32)
    >>> fh.close()

When writing to file, we again need to pass in ``sample_rate`` and ``nchan``,
though time can either be passed explicitly or inferred from the header::


    >>> fw = mark5b.open('test.m5b', 'ws', header0=header0,
    ...                  sample_rate=32*u.MHz, nchan=8)
    >>> fw.write(d)
    >>> fw.close()
    >>> fh = mark5b.open('test.m5b', 'rs', sample_rate=32*u.MHz,
    ...                  kday=57000, nchan=8)
    >>> np.all(fh.read() == d)
    True
    >>> fh.close()

.. _mark5b_api:

Reference/API
=============

.. automodapi:: baseband.mark5b
.. automodapi:: baseband.mark5b.header
   :include-all-objects:
.. automodapi:: baseband.mark5b.payload
.. automodapi:: baseband.mark5b.frame
.. automodapi:: baseband.mark5b.file_info
.. automodapi:: baseband.mark5b.base
