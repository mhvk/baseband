"""__      __  _____    _   _____
 \ \    / / | ___ \  | | |   __|
  \ \  / /  | |  | | | | |  |_
   \ \/ /   | |  | | | | |   _]
    \  /    | |__| | | | |  |
     \/     |_____/  |_| |__|

VDIF VLBI format readers, providing both low-level and higher-level access.

All files should be opened using ``vdif.open``.  Opening in binary mode
provides a normal file reader but extended with methods to read a VDIF frame
(or sets of frames):

>>> from scintellometry.io import vdif
>>> fh = vdif.open('vlba.m5a', 'rb')
>>> fs = fh.read_frameset()
>>> fs.data.shape
(8, 20000, 1)


Opening in stream mode wraps the low-level routines such that reading
and writing is in units of samples.  It also provides access to header
information.

>>> fh = vdif.open('vlba.m5a', 'rs')
>>> fh
<VDIFStreamReader name=vlba.m5a offset=0
    nthread=2, samples_per_frame=20000, nchan=1,
    station=65534, (start) time=2014-06-13T05:30:00.000000000,
    bandwidth=16.0 MHz, complex_data=False, bps=2, edv=3>

>>> d = fh.read(12)
>>> d.shape
(12, 8)
>>> d[:, 0].astype(int)  # first thread
array([ 3, -1,  1,  1, -3, -1,  3,  3,  1,  1,  1,  3])

One can pick specific threads:
>>> fh = vdif.open('vlba.m5a', 'rs', thread_ids=[2, 3])
>>> d = fh.read(20000)
>>> d.shape
(20000, 2)

To set up a file for writing needs quite a bit of header information. Not
coincidentally, what is given by the reader above suffices:

>>> from astropy.time import Time
>>> import astropy.units as u
>>> fw = vdif.open('try.vdif', 'ws',
...                nthread=2, samples_per_frame=20000, nchan=1,
...                station=65534, time=Time('2014-06-13T05:30:00.000000000'),
...                bandwidth=16.0*u.MHz, complex_data=False, bps=2, edv=3)
>>> fw.write(d)
>>> fw.close()
>>> fh = vdif.open('try.vdif', 'rs')
>>> d2 = fh.read(20000)
>>> np.all(d == d2)
True


Example to copy a VDIF file.  The data should be identical, though frames
will be ordered by thread_id.  (This can be avoided by using ``read_frame``
 and ``write_frame``, resp., but then, of course, copying can be done easier!)

>>> from scintellometry.io import vdif
>>> with vdif.open('vlba.m5a', 'rb') as fr, vdif.open('try.vdif', 'wb') as fw:
...     while(True):
...         try:
...             fw.write_frameset(fr.read_frameset())
...         except:
...             break

For small files, one could just do:
>>> with vdif.open('vlba.m5a', 'rs') as fr, vdif.open(
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
from .data import VDIFData
