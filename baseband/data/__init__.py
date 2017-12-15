# Licensed under the GPLv3 - see LICENSE.rst
"""Sample files with baseband data recorded in different formats."""

# Use private names to avoid inclusion in the sphinx documentation.
from os import path as _path


def _full_path(name, dirname=_path.dirname(_path.abspath(__file__))):
    return _path.join(dirname, name)


SAMPLE_MARK4 = _full_path('sample.m4')
"""Mark 4 sample.  ntrack=64, fanout=4, bps=2.

Created from a European VLBI Network/Arecibo PSR B1957+20 observation using
dd if=gp052d_ar_no0021 of=sample.m4 bs=128000 count=3
"""

SAMPLE_MARK4_32TRACK = _full_path('sample_32track.m4')
"""Mark 4 sample.  ntrack=32, fanout=4, bps=2.

Created from a Arecibo observation simultaneous with RadioAstron using
dd if=rg10a_ar_no0014 of=sample_32track.m4 bs=10000 count=17
"""

SAMPLE_MARK4_32TRACK_FANOUT2 = _full_path('sample_32track_fanout2.m4')
"""Mark 4 sample.  ntrack=32, fanout=2, bps=2.

Created from an Arecibo observation of PSR B1133+16 using
dd if=gk049c_ar_no0011.m5a of=sample_32track_fanout2.m4 bs=10000 count=18
"""

SAMPLE_MARK4_16TRACK = _full_path('sample_16track.m4')
"""Mark 4 sample.  ntrack=16, fanout=4, bps=2.

Created from the first two frames an Arecibo observation of the Crab Pulsar on
2013/11/03.  (2013_306_raks02ae/ar/gs033a_ar_no0055.m5a)
"""

SAMPLE_MARK5B = _full_path('sample.m5b')
"""Mark 5B sample.  nchan=8, bps=2.

Created from a EVN/WSRT PSR B1957+20 observation.
"""

SAMPLE_VDIF = _full_path('sample.vdif')
"""VDIF sample. 8 threads, bps=2.

Created from a EVN/VLBA PSR B1957+20 observation.
"""

SAMPLE_MWA_VDIF = _full_path('sample_mwa.vdif')
"""VDIF sample from MWA.  EDV=0, two threads, bps=8"""

SAMPLE_AROCHIME_VDIF = _full_path('sample_arochime.vdif')
"""VDIF sample from ARO, written by CHIME backend. EDV=1, nchan=1024, bps=4."""

SAMPLE_DADA = _full_path('sample.dada')
"""DADA sample from Effelsberg, with header adapted to shortened size."""

SAMPLE_GSB_HEADER = _full_path('sample_gsb.timestamp')
"""GSB header sample from 2013/07/26 GMRT observations of PSR J1810+1744,
adapted to shortened size.
"""

SAMPLE_DRAO_CORRUPT = _full_path('sample_drao_corrupted.vdif')
"""Corrupted VDIF sample. bps=4.

First ten frames extracted from b0329 DRAO corrupted raw data file
0059000.dat.
"""
