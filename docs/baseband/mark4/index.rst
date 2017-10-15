.. _mark4:

******
MARK 4
******

The Mark 4 VLBI format is described in the `Mark IIIA/IV/VLBA documentation
<http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf>`_.

.. _mark4_usage:

Usage
=====

All files should be opened using :func:`~baseband.mark4.open`.  Opening in
binary mode provides a normal file reader but extended with methods to read a
:class:`~baseband.mark4.Mark4Frame`.  For Mark 4 data, the files often do not
start at a frame boundary, so one has to seek the first frame.  One also has
to pass in the number of tracks used, and the decade the data were taken, since
those numbers cannot be inferred from the data themselves::

    >>> from baseband import mark4
    >>> from baseband.data import SAMPLE_MARK4
    >>> fh = mark4.open(SAMPLE_MARK4, 'rb')
    >>> fh.find_frame(ntrack=64)
    2696
    >>> frame = fh.read_frame(ntrack=64, decade=2010)
    >>> frame.shape
    (80000, 8)
    >>> fh.close()


Opening in stream mode wraps the low-level routines such that reading and
writing is in units of samples.  It also provides access to header information.
Here, we need to pass in the frame rate so that times for individual samples
can be calculated (for longer files, this can be calculated from the file)::

    >>> fh = mark4.open(SAMPLE_MARK4, 'rs', ntrack=64, decade=2010,
    ...                 frames_per_second=400)
    >>> fh
    <Mark4StreamReader name=... offset=0
        frames_per_second=400, samples_per_frame=80000,
        sample_shape=SampleShape(nchan=8), bps=2,
        (start) time=2014-06-16T07:38:12.47500>
    >>> d = fh.read(6400)
    >>> d.shape
    (6400, 8)
    >>> d[635:645, 0].astype(int)  # first thread
    array([ 0,  0,  0,  0,  0, -1,  1,  3,  1, -1])
    >>> fh.close()

Note that the first 640 elements of every frame are set to zero, as those data
were overwritten by the header.

To set up a file for writing as a stream is possible as well, although
probably one would prefer to use VDIF file instead.  Here, we set the number of
samples per frame to 640, so that the invalid parts of the mark4 data are all
captured in one frame::

    >>> from astropy.time import Time
    >>> import astropy.units as u, numpy as np
    >>> from baseband import vdif
    >>> fw = vdif.open('try.vdif', 'ws',
    ...                nthread=1, samples_per_frame=640, nchan=8,
    ...                framerate=50000*u.Hz, complex_data=False, bps=2, edv=1,
    ...                time=Time('2014-06-16T07:38:12.47500'))
    >>> fw.write(d[:640], invalid_data=True)
    >>> fw.write(d[640:])
    >>> fw.close()
    >>> fr = vdif.open('try.vdif', 'rs')
    >>> d2 = fr.read()
    >>> np.all(d == d2)
    True
    >>> fr.close()

.. _mark4_api:

Reference/API
=============

.. automodapi:: baseband.mark4
.. automodapi:: baseband.mark4.header
   :include-all-objects:
.. automodapi:: baseband.mark4.payload
.. automodapi:: baseband.mark4.frame
.. automodapi:: baseband.mark4.base
