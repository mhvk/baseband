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

These data follow standard assignments:
fan_out       0011223300112233001122330011223300112233001122330011223300112233
magnitude_bit 0000000011111111000000001111111100000000111111110000000011111111
lsb_output    1111111111111111111111111111111111111111111111111111111111111111
converter_id  0202020202020202131313131313131346464646464646465757575757575757
"""

SAMPLE_MARK4_32TRACK = _full_path('sample_32track.m4')
"""Mark 4 sample.  ntrack=32, fanout=4, bps=2.

Created from a Arecibo observation simultaneous with RadioAstron using
dd if=rg10a_ar_no0014 of=sample_32track.m4 bs=10000 count=17

These data follow standard assignments:
fan_out       00112233001122330011223300112233
magnitude_bit 00000000111111110000000011111111
lsb_output    01010101010101010101010101010101
converted_id  00000000000000001111111111111111
"""

SAMPLE_MARK4_32TRACK_FANOUT2 = _full_path('sample_32track_fanout2.m4')
"""Mark 4 sample.  ntrack=32, fanout=2, bps=2.

Created from an Arecibo observation of PSR B1133+16 using
dd if=gk049c_ar_no0011.m5a of=sample_32track_fanout2.m4 bs=10000 count=18

These data follow standard assignments:
fan_out       00110011001100110011001100110011
magnitude_bit 00001111000011110000111100001111
lsb_output    00000000000000001111111111111111
converted_id  02020202131313130202020213131313
"""

SAMPLE_MARK4_16TRACK = _full_path('sample_16track.m4')
"""Mark 4 sample.  ntrack=16, fanout=4, bps=2.

Created from the first two frames an Arecibo observation of the Crab Pulsar on
2013/11/03.  (2013_306_raks02ae/ar/gs033a_ar_no0055.m5a)

These data follow standard assignments:
fan_out       0123012301230123
magnitude_bit 0000111100001111
lsb_output    1111111111111111
converted_id  0000000011111111
"""

SAMPLE_MARK4_64TRACK_FANOUT2_FT = _full_path('sample_64track_fanout2_ft.m4')
"""Mark 4 sample. ntrack=64, fanout=2, bps=2.

Created from Fortleza (Ft) station by Shaogang Gao (gh-332).

These data do not follow standard assignments, so need special treatment:
fan_out       0011001100110011001100110011001100110011001100110011001100110011
magnitude_bit 0000010110101111000011110000111100000101101011110000111100001111
lsb_output    0000101000001010000000000000000000001010000010100000000000000000
converted_id  030303030404040415151515262626267a7a7a7a7b7b7b7b8c8c8c8c9d9d9d9d
"""

SAMPLE_MARK5B = _full_path('sample.m5b')
"""Mark 5B sample.  nchan=8, bps=2.

Created from a EVN/WSRT PSR B1957+20 observation.
"""

SAMPLE_VDIF = _full_path('sample.vdif')
"""VDIF sample. 8 threads, bps=2.

Created from a EVN/VLBA PSR B1957+20 observation.  Timestamps of frames with
even thread IDs have been corrected to be consistent with odd-ID frames.
"""

SAMPLE_VLBI_VDIF = _full_path('sample_vlbi.vdif')
"""VDIF sample. 8 threads, bps=2.

Created from a EVN/VLBA PSR B1957+20 observation.  Uncorrected version of
SAMPLE_VDIF.
"""

SAMPLE_MWA_VDIF = _full_path('sample_mwa.vdif')
"""VDIF sample from MWA.  EDV=0, two threads, bps=8"""

SAMPLE_AROCHIME_VDIF = _full_path('sample_arochime.vdif')
"""VDIF sample from ARO, written by CHIME backend. EDV=1, nchan=1024, bps=4."""

SAMPLE_BPS1_VDIF = _full_path('sample_bps1.vdif')
"""VDIF sample from Christian Ploetz. EDV=0, nchan=16, bps=1."""

SAMPLE_DADA = _full_path('sample.dada')
"""DADA sample from Effelsberg, with header adapted to shortened size."""

SAMPLE_PUPPI = _full_path('sample_puppi.raw')
"""GUPPI/PUPPI sample, npol=2, nchan=4.

Created from the first four frames of a 2018-01-14 Arecibo observation of
J1810+1744, with payload shortened to 8192 samples (with 512 overlap),
equivalent to 1024 complete samples (with 64 overlap).
"""

SAMPLE_GSB_RAWDUMP_HEADER = _full_path('gsb/sample_gsb_rawdump.timestamp')
"""GSB rawdump header sample.

First 10 header entries of node 5 rawdump data from 2015-04-27 GMRT
observations of the Crab pulsar.
"""

SAMPLE_GSB_RAWDUMP = _full_path('gsb/sample_gsb_rawdump.dat')
"""GSB rawdump sample.  samples_per_frame=8192

First 81920 samples of node 5 rawdump data from 2015-04-27 GMRT observations of
the Crab pulsar.
"""

SAMPLE_GSB_PHASED_HEADER = _full_path('gsb/sample_gsb_phased.timestamp')
"""GSB phased header sample.

10 header entries, starting from seq_nr=9994, from 2013-07-27 GMRT observations
of PSR J1810+1744.
"""

SAMPLE_GSB_PHASED = ((_full_path('gsb/sample_gsb_phased.Pol-L1.dat'),
                      _full_path('gsb/sample_gsb_phased.Pol-L2.dat')),
                     (_full_path('gsb/sample_gsb_phased.Pol-R1.dat'),
                      _full_path('gsb/sample_gsb_phased.Pol-R2.dat')))
"""GSB phased sample.  samples_per_frame=8

80 complete samples, starting from seq_nr=9994, from 2013-07-27 GMRT
observations of PSR J1810+1744, rewritten so each frame has 8 complete samples.
"""

SAMPLE_DRAO_CORRUPT = _full_path('sample_drao_corrupted.vdif')
"""Corrupted VDIF sample. bps=4.

First ten frames extracted from b0329 DRAO corrupted raw data file
0059000.dat.
"""
