# Licensed under the GPLv3 - see LICENSE.rst
"""Sample files with baseband data recorded in different formats."""

import os


def full_path(name, dirname=os.path.dirname(os.path.abspath(__file__))):
    return os.path.join(dirname, name)


SAMPLE_MARK4 = full_path('sample.m4')
"""Mark 4 sample.  ntrack=64, fanout=4, bps=2.

Created from a European VLBI Network/Arecibo PSR B1957+20 observation using
dd if=gp052d_ar_no0021 of=sample.m4 bs=128000 count=3
"""

SAMPLE_MARK4_32TRACK = full_path('sample_32track.m4')
"""Mark 4 sample.  ntrack=32, fanout=4, bps=2.

Created from a Arecibo observation simultaneous with RadioAstron using
dd if=ar/rg10a_ar_no0014 of=sample_32track.m4 bs=10000 count=17
"""

SAMPLE_MARK5B = full_path('sample.m5b')
"""Mark 5B sample.  nchan=8, bps=2.

Created from a EVN/WSRT PSR B1957+20 observation.
"""

SAMPLE_VDIF = full_path('sample.vdif')
"""VDIF sample. 8 threads, bps=2.

Created from a EVN/VLBA PSR B1957+20 observation.
"""

SAMPLE_MWA_VDIF = full_path('sample_mwa.vdif')
"""VDIF sample from MWA.  EDV=0, two threads, bps=8"""

SAMPLE_AROCHIME_VDIF = full_path('sample_arochime.vdif')
"""VDIF sample from ARO, written by CHIME backend. EDV=1, nchan=1024, bps=4."""

SAMPLE_DADA = full_path('sample.dada')
"""DADA sample from Effelsberg, with header adapted to shortened size."""

# Clean up namespace
del os
del full_path
