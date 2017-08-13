# Licensed under the GPLv3 - see LICENSE.rst
#
# __      __  _____    _   _____
# \ \    / / | ___ \  | | |   __|
#  \ \  / /  | |  | | | | |  |_
#   \ \/ /   | |  | | | | |   _]
#    \  /    | |__| | | | |  |
#     \/     |_____/  |_| |__|
#
#
"""VDIF VLBI format readers, providing both low-level and higher-level access.

All files should be opened using :func:`~baseband.vdif.open`.  Opening in
binary mode provides a normal file reader but extended with methods to read a
:class:`~baseband.vdif.VDIFFrame` (or :class:`~baseband.vdif.FrameSet`):

>>> from baseband import vdif
>>> from baseband.data import SAMPLE_VDIF
>>> fh = vdif.open(SAMPLE_VDIF, 'rb')
>>> fs = fh.read_frameset()
>>> fs.data.shape
(8, 20000, 1)
>>> fh.close()

Opening in stream mode wraps the low-level routines such that reading
and writing is in units of samples.  It also provides access to header
information.

>>> fh = vdif.open(SAMPLE_VDIF, 'rs')
>>> fh
<VDIFStreamReader name=... offset=0
    nthread=8, samples_per_frame=20000, nchan=1,
    frames_per_second=1600, complex_data=False, bps=2, edv=3,
    station=65532, (start) time=2014-06-16T05:56:07.000000000>
>>> d = fh.read(12)
>>> d.shape
(12, 8)
>>> d[:, 0].astype(int)  # first thread
array([-1, -1,  3, -1,  1, -1,  3, -1,  1,  3, -1,  1])
>>> fh.close()

One can pick specific threads:

>>> fh = vdif.open(SAMPLE_VDIF, 'rs', thread_ids=[2, 3])
>>> d = fh.read(20000)
>>> d.shape
(20000, 2)
>>> fh.close()

To set up a file for writing needs quite a bit of header information. Not
coincidentally, what is given by the reader above suffices:

>>> from astropy.time import Time
>>> import astropy.units as u, numpy as np
>>> fw = vdif.open('try.vdif', 'ws',
...                nthread=2, samples_per_frame=20000, nchan=1,
...                frames_per_second=1600, complex_data=False, bps=2, edv=3,
...                station=65532, time=Time('2014-06-16T05:56:07.000000000'))
>>> fw.write(d)
>>> fw.close()
>>> fh = vdif.open('try.vdif', 'rs')
>>> d2 = fh.read(12)
>>> np.all(d[:12] == d2)
True
>>> fh.close()

Example to copy a VDIF file.  Here, we use the ``sort=False`` option to ensure
the frames are written exactly in the same order, so the files should be
identical.

>>> with vdif.open(SAMPLE_VDIF, 'rb') as fr, vdif.open('try.vdif', 'wb') as fw:
...     while True:
...         try:
...             fw.write_frameset(fr.read_frameset(sort=False))
...         except:
...             break

For small files, one could just do:

>>> with vdif.open(SAMPLE_VDIF, 'rs') as fr, vdif.open(
...         'try.vdif', 'ws', header=fr.header0, nthread=fr.nthread) as fw:
...     fw.write(fr.read())

This copies everything to memory, though, and some header information is lost.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from .base import open
from .header import VDIFHeader
from .payload import VDIFPayload
from .frame import VDIFFrame, VDIFFrameSet
