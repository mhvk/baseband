# Licensed under the GPLv3 - see LICENSE.rst
"""DADA baseband data reader.

All files should be opened using :func:`~baseband.dada.open`.  Single files can
be opened in binary mode, which provides a normal file reader but extended with
methods to read a :class:`~baseband.dada.DADAFrame`.  For dada files, which
consist of just a single header and payload, such frames contain all the data.

>>> from baseband import dada
>>> from baseband.data import SAMPLE_DADA
>>> fh = dada.open(SAMPLE_DADA, 'rb')
>>> frame = fh.read_frame()
>>> frame.shape
(16000, 2, 1)
>>> frame[:3].squeeze()
array([[ -38.-38.j,  -38.-38.j],
       [ -38.-38.j,  -40. +0.j],
       [-105.+60.j,   85.-15.j]], dtype=complex64)

Since the files can be quite large, the payload is mapped, so that if one
accesses part of the data, only the corresponding parts of the encoded payload
are loaded into memory (since the sample file is encoded using 8 bits, the
above example thus loads 12 bytes into memory).

Opening in stream mode wraps the low-level routines such that reading and
writing is in units of samples, and one has access to header information.

>>> fh = dada.open(SAMPLE_DADA, 'rs')
>>> fh
<DADAStreamReader name=... offset=0
    samples_per_frame=16000, nchan=1, frames_per_second=1000.0, bps=8,
    thread_ids=[0, 1], (start) time=2013-07-02T01:39:20.000>
>>> d = fh.read(10000)
>>> d.shape
(10000, 2)
>>> d[:3]  # first thread
array([[ -38.-38.j,  -38.-38.j],
       [ -38.-38.j,  -40. +0.j],
       [-105.+60.j,   85.-15.j]], dtype=complex64)

To set up a file for writing as a stream is possible as well.  Here, we use an
even smaller size of the payload, to show how one can define multiple files.

>>> from astropy.time import Time
>>> import astropy.units as u
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
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from .base import open
from .header import DADAHeader
from .payload import DADAPayload
from .frame import DADAFrame
