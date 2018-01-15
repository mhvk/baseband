.. _mark4:

******
MARK 4
******

The Mark 4 VLBI format is described in the `Mark IIIA/IV/VLBA documentation
<http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf>`_.

.. _mark4_usage:

Usage
=====

This section covers Mark 4-specific features of Baseband.  Tutorials for general
usage can be found under the :ref:`Using Baseband <using_baseband_toc>` section.
The examples below use the small sample file ``baseband/data/sample.m4``,
and assumes `numpy` and `baseband.mark4` modules have been imported::

    >>> import numpy as np
    >>> from baseband import mark4
    >>> from baseband.data import SAMPLE_MARK4

Opening a Mark 4 file with :func:`~baseband.mark4.open` in binary mode provides
a normal file reader but extended with methods to read a
:class:`~baseband.mark4.Mark4Frame`.  Mark 4 data files generally do not start 
(or end) at a frame boundary, so in binary mode one has to seek the first frame
using `~baseband.mark4.base.Mark4StreamReader.find_frame`.  One also has to pass
in the number of tracks used to `~baseband.mark4.base.Mark4StreamReader.read_frame`,
and the decade the data were taken, since those numbers cannot be inferred from
the data themselves::

    >>> fh = mark4.open(SAMPLE_MARK4, 'rb')
    >>> fh.find_frame(ntrack=64)    # Find first frame.
    2696
    >>> frame = fh.read_frame(ntrack=64, decade=2010)
    >>> frame.shape
    (80000, 8)
    >>> fh.close()

Opening in stream mode automatically seeks for the first frame, and wraps the
low-level routines such that reading and writing is in units of samples.  It
also provides access to header information.  In lieu of ``decade``, one may
also provide a reference time within 4 years of the observation start time::

    >>> from astropy.time import Time
    >>> fh = mark4.open(SAMPLE_MARK4, 'rs', ntrack=64,
    ...                 ref_time=Time('2013:100:23:00:00'))
    >>> fh
    <Mark4StreamReader name=... offset=0
        frames_per_second=400, samples_per_frame=80000,
        sample_shape=SampleShape(nchan=8), bps=2,
        start_time=2014-06-16T07:38:12.47500>
    >>> d = fh.read(6400)
    >>> d.shape
    (6400, 8)
    >>> d[635:645, 0].astype(int)  # first thread
    array([ 0,  0,  0,  0,  0, -1,  1,  3,  1, -1])
    >>> fh.close()

For Mark 4 files, the header takes the place of the first 160 samples of each
track, such that the first payload sample occurs ''fanout * 160'' sample times
after the header time.  The stream reader includes these overwritten samples as
invalid data (zeros, by default)::

    >>> np.array_equal(d[:640], np.zeros((640,) + d.shape[1:]))
    True

When writing to file, we need to pass in the frame rate in addition to 
``ntrack`` and ``decade`` so that times for individual samples can be
calculated.

    >>> fw = mark4.open('sample_mark4_segment.m4', 'ws', header=frame.header,
    ...                 ntrack=64, decade=2010, frames_per_second=400)
    >>> fw.write(frame.data)
    >>> fw.close()

.. _mark4_api:

Reference/API
=============

.. automodapi:: baseband.mark4
.. automodapi:: baseband.mark4.header
   :include-all-objects:
.. automodapi:: baseband.mark4.payload
.. automodapi:: baseband.mark4.frame
.. automodapi:: baseband.mark4.base
