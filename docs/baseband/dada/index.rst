.. _dada:

****
DADA
****

Distributed Acquisition and Data Analysis (DADA) format data files contain an
ASCII header of typically 4096 bytes followed by a payload.

.. _dada_usage:

Usage
=====

This section covers DADA-specific features of Baseband.  Tutorials for general
usage can be found under the :ref:`Using Baseband <using_baseband_toc>` section.
The examples below use the small sample file ``baseband/data/sample.dada``,
and assume the `astropy.units` and `baseband.dada` modules have been imported::

    >>> from baseband import dada
    >>> import astropy.units as u
    >>> from baseband.data import SAMPLE_DADA

Single files can be opened with :func:`~baseband.dada.open` in binary mode. 
Dada files consist of just a single header and payload, and can be read into a
single :class:`~baseband.dada.DADAFrame`.

::

    >>> fh = dada.open(SAMPLE_DADA, 'rb')
    >>> frame = fh.read_frame()
    >>> frame.shape
    (16000, 2, 1)
    >>> frame[:3].squeeze()
    array([[ -38.-38.j,  -38.-38.j],
           [ -38.-38.j,  -40. +0.j],
           [-105.+60.j,   85.-15.j]], dtype=complex64)
    >>> fh.close()

Since the files can be quite large, the payload is mapped, so that if one
accesses part of the data, only the corresponding parts of the encoded payload
are loaded into memory (since the sample file is encoded using 8 bits, the
above example thus loads 12 bytes into memory).

Opening in stream mode wraps the low-level routines such that reading and
writing is in units of samples, and provides access to header information.

::

    >>> fh = dada.open(SAMPLE_DADA, 'rs')
    >>> fh
    <DADAStreamReader name=... offset=0
        sample_rate=1.6e+07 Hz, samples_per_frame=16000,
        sample_shape=SampleShape(npol=2), bps=8,
        thread_ids=[0, 1], start_time=2013-07-02T01:39:20.000>
    >>> d = fh.read(10000)
    >>> d.shape
    (10000, 2)
    >>> d[:3]
    array([[ -38.-38.j,  -38.-38.j],
           [ -38.-38.j,  -40. +0.j],
           [-105.+60.j,   85.-15.j]], dtype=complex64)
    >>> fh.close()

To set up a file for writing as a stream is possible as well.  Here, we use an
even smaller size of the payload, to show how one can define multiple files.

::

    >>> from astropy.time import Time
    >>> fw = dada.open('{utc_start}.{obs_offset:016d}.000000.dada', 'ws',
    ...                npol=2, samples_per_frame=5000, nchan=1, bps=8,
    ...                bandwidth=16*u.MHz, complex_data=True,
    ...                time=Time('2013-07-02T01:39:20.000'))
    >>> fw.write(d)
    >>> fw.close()
    >>> import os
    >>> [f for f in sorted(os.listdir('.')) if f.startswith('2013')]
    ['2013-07-02-01:39:20.0000000000000000.000000.dada',
     '2013-07-02-01:39:20.0000000000020000.000000.dada']
    >>> fr = dada.open('2013-07-02-01:39:20.{obs_offset:016d}.000000.dada', 'rs')
    >>> d2 = fr.read()
    >>> (d == d2).all()
    True
    >>> fr.close()

.. _dada_api:

Reference/API
=============

.. automodapi:: baseband.dada
.. automodapi:: baseband.dada.header
.. automodapi:: baseband.dada.payload
.. automodapi:: baseband.dada.frame
.. automodapi:: baseband.dada.base
