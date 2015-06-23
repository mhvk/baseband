"""Mark5B VLBI data reader.  Code inspired by Walter Brisken's mark5access.
See https://github.com/demorest/mark5access.
Also, for the Mark5B design, see
http://www.haystack.mit.edu/tech/vlbi/mark5/mark5_memos/019.pdf
"""
from __future__ import division, unicode_literals
import os
import warnings

import numpy as np
from astropy.utils.compat.odict import OrderedDict
from astropy.time import Time, TimeDelta
import astropy.units as u

from . import SequentialFile, header_defaults
from .vlbi_helpers import (make_parser, bcd_decode, get_frame_rate,
                           four_word_struct, OPTIMAL_2BIT_HIGH)


# the high mag value for 2-bit reconstruction
SYNC_PATTERN = 0xABADDEED
PAYLOADSIZE = 2500 * 4  # 2500 words
HEADERSIZE = 4 * 4  # 4 words


# Check code on 2015-MAY-08.
# m5d /raw/mhvk/scintillometry/gp052d_wb_no0001 Mark5B-512-8-2 10
# Mark5 stream: 0x256d140
#   stream = File-1/1=gp052a_wb_no0001
#   format = Mark5B-512-8-2 = 2
#   start mjd/sec = 821 19801.000000000
#   frame duration = 156250.00 ns
#   framenum = 0
#   sample rate = 32000000 Hz
#   offset = 0
#   framebytes = 10016 bytes
#   datasize = 10000 bytes
#   sample granularity = 1
#   frame granularity = 1
#   gframens = 156250
#   payload offset = 16
#   read position = 0
#   data window size = 1048576 bytes
# -3 -1  1 -1  3 -3 -3  3
# -3  3 -1  3 -1 -1 -1  1
#  3 -1  3  3  1 -1  3 -1
# Compare with my code:
# fh = Mark5BData(['/raw/mhvk/scintillometry/gp052d_wb_no0001'],
#                  channels=None, fedge=0, fedge_at_top=True)
# 'Start time: ', '2014-06-13 05:30:01.000' -> correct
# fh.header0
# <Mark5BFrameHeader sync_pattern: 0xabaddeed,
#                    year: 11,
#                    user: 3757,
#                    internal_tvg: False,
#                    frame_nr: 0,
#                    bcd_jday: 0x821,
#                    bcd_seconds: 0x19801,
#                    bcd_fraction: 0x0,
#                    crcc: 38749>
# fh.record_read(6).astype(int)
# array([[-3, -1,  1, -1,  3, -3, -3,  3],
#        [-3,  3, -1,  3, -1, -1, -1,  1],
#        [ 3, -1,  3,  3,  1, -1,  3, -1]])


class Mark5BData(SequentialFile):

    telescope = 'mark5b'
    payloadsize = PAYLOADSIZE

    def __init__(self, raw_files, channels, fedge, fedge_at_top,
                 blocksize=None, Mbps=512, nvlbichan=8, nbit=2,
                 decimation=1, reftime=Time('J2010.', scale='utc'), comm=None):
        """Mark 4 Data reader.

        Parameters
        ----------
        raw_files : list of string
            full file names of the Mark 4 data
        channels : list of int
            channel numbers to read; should be at the same frequency,
            i.e., 1 or 2 polarisations.
        fedge : Quantity
            Frequency at the edge of the requested VLBI channel
        fedge_at_top : bool
            Whether the frequency is at the top of the channel.
        blocksize : int or None
            Number of bytes typically read in one go (default: framesize).
        Mbps : Quantity
            Total bit rate.  Only used to check consistency with the data.
        nvlbichan : int
            Number of VLBI channels encoded in Mark 4 data stream.
        nbit : int
            Number of bits per sample.
        decimation : int
            Number by which the samples should be decimated (default: 1, i.e.,
            no decimation).
        reftime : `~astropy.time.Time` instance
            Time close(ish) to the observation time, to resolve decade
            ambiguities in the times stored in the Mark 4 data frames.
        comm : MPI communicator
            For consistency with other readers; not used in this one.
        """
        assert nbit == 1 or nbit == 2
        assert decimation == 1 or decimation == 2 or decimation % 4 == 0
        nbitstream = nvlbichan * nbit
        assert nbitstream in (1, 2, 4, 8, 16, 32)
        self.Mbps = u.Quantity(Mbps, u.Mbit / u.s)
        self.nvlbichan = nvlbichan
        self.nbit = nbit
        self.decimation = decimation
        self.nbitstream = nbitstream
        self.fedge = fedge
        self.fedge_at_top = fedge_at_top
        # assert 1 <= len(channels) <= 2
        self.channels = channels
        try:
            self.npol = len(channels)
        except TypeError:
            self.npol = self.nvlbichan if channels is None else 1
        if not (1 <= self.npol <= 2):
            warnings.warn("Should use 1 or 2 channels for folding!")
        # PAYLOADSIZE and HEADERSIZE are fixed for Mark 5B
        self.framesize = PAYLOADSIZE + HEADERSIZE
        # Comment from C code:
        # /* YES: the following is a negative number.  This is OK because the
        #    mark4 blanker will prevent access before element 0. */
        # The offset will also be relative to a positive frame position.
        self._decode = DECODERS[self.nbit]
        # Initialize standard reader, setting self.files, self.blocksize,
        # dtype, nchan, itemsize, recordsize, setsize.
        if blocksize is None:
            blocksize = self.framesize
        dtype = '{0:d}u1'.format(self.nbitstream // 8)
        self.filesize = os.path.getsize(raw_files[0])
        super(Mark5BData, self).__init__(raw_files, blocksize=blocksize,
                                         dtype=dtype, nchan=1, comm=comm)
        # Above also opened first file, so use it now to determine
        # the time of the first frame, which is also the start time.
        # Just ensure we're fine.
        self.header0 = Mark5BFrameHeader.fromfile(self.fh_raw)
        self.time0 = self.header0.time()
        assert abs(self.header0.time().mjd - reftime.mjd) < 3650
        frame_rate = get_frame_rate(self.fh_raw, Mark5BFrameHeader) * u.Hz
        self.samplerate = ((PAYLOADSIZE // self.nvlbichan * 8 // self.nbit) *
                           frame_rate).to(u.MHz)
        self.dtsample = (1. / self.samplerate).to(u.ns)
        self.seek(0)
        # Check that the Mbps passed in is consistent with this file.
        mbps_est = self.samplerate * self.nvlbichan * self.nbit * u.bit
        if mbps_est != self.Mbps:
            warnings.warn("Warning: the data rate passed in ({0}) disagrees "
                          "with that calculated ({1})."
                          .format(self.Mbps, mbps_est))

        if comm is None or comm.rank == 0:
            print("In MARK5BData, done initialising")
            print("Start time: ", self.time0.iso)

    def _seek(self, offset):
        assert offset % self.recordsize == 0
        # Seek in the raw file using framesize, i.e., including headers.
        self.fh_raw.seek(offset // self.payloadsize * self.framesize)
        self.offset = offset

    def record_read(self, count):
        """Read and decode count bytes.

        The range retrieved can span multiple frames and files.

        Parameters
        ----------
        count : int
            Number of bytes to read.

        Returns
        -------
        data : array of float
            Dimensions are [sample-time, vlbi-channel].
        """
        # for now only allow integer number of frames
        assert count % self.recordsize == 0
        data = np.empty((count // self.recordsize, self.npol),
                        dtype=np.float32)
        sample = 0
        # With the payloadoffset applied, as we do, the invalid part from
        # VALIDEND to PAYLOADSIZE is also at the start.  Thus, the total size
        # at the start is this one plus the part before VALIDSTART.
        while count > 0:
            # Validate frame we're reading from.
            payload, payload_offset = divmod(self.offset, self.payloadsize)
            self.seek(payload * self.payloadsize)
            to_read = min(count, self.payloadsize - payload_offset)
            # read header and verify it.
            header = Mark5BFrameHeader.fromfile(self.fh_raw, verify=True)
            # ADD CHECK ON header['frame_nr']
            # this leaves raw_file pointer at start of payload.
            if payload_offset > 0:
                self.fh_raw.seek(payload_offset, 1)

            raw = np.fromstring(self.fh_raw.read(to_read), np.uint8)
            nsample = len(raw) // self.recordsize
            data[sample:sample + nsample] = self._decode(raw, self.nvlbichan,
                                                         self.channels)

            self.offset += to_read
            sample += nsample
            count -= to_read

        # ensure offset pointers from raw and virtual match again,
        # and are at the end of what has been read.
        if self.npol == 2:
            data = data.view('{0},{0}'.format(data.dtype.str))

        return data

    # def _seek(self, offset):
    #     assert offset % self.recordsize == 0
    #     # Find the correct file.
    #     file_number = np.searchsorted(self.payloadranges, offset)
    #     self.open(self.files, file_number)
    #     if file_number > 0:
    #         file_offset = offset
    #     else:
    #         file_offset = offset - self.payloadranges[file_number - 1]
    #     # Find the correct frame within the file.
    #     frame_nr, frame_offset = divmod(file_offset, self.payloadsize)
    #     self.fh_raw.seek(frame_nr * self.framesize + self.header_size)
    #     self.offset = offset

    def __str__(self):
        return ('<Mark5BData nvlbichan={0} nbit={1} dtype={2} blocksize={3}\n'
                'current_file_number={4}/{5} current_file={6}>'
                .format(self.nvlbichan, self.nbit, self.dtype, self.blocksize,
                        self.current_file_number, len(self.files),
                        self.files[self.current_file_number]))


# VDIF defaults for psrfits HDUs
# Note: these are largely made-up at this point
header_defaults['mark5b'] = {
    'PRIMARY': {'TELESCOP':'MARK5B',
                'IBEAM':1, 'FD_POLN':'LIN',
                'OBS_MODE':'SEARCH',
                'ANT_X':0, 'ANT_Y':0, 'ANT_Z':0, 'NRCVR':1,
                'FD_HAND':1, 'FD_SANG':0, 'FD_XYPH':0,
                'BE_PHASE':0, 'BE_DCC':0, 'BE_DELAY':0,
                'TRK_MODE':'TRACK',
                'TCYCLE':0, 'OBSFREQ':300, 'OBSBW':100,
                'OBSNCHAN':0, 'CHAN_DM':0,
                'EQUINOX':2000.0, 'BMAJ':1, 'BMIN':1, 'BPA':0,
                'SCANLEN':1, 'FA_REQ':0,
                'CAL_FREQ':0, 'CAL_DCYC':0, 'CAL_PHS':0, 'CAL_NPHS':0,
                'STT_IMJD':54000, 'STT_SMJD':0, 'STT_OFFS':0},
    'SUBINT': {'INT_TYPE': 'TIME',
               'SCALE': 'FluxDen',
               'POL_TYPE': 'AABB',
               'NPOL':1,
               'NBIN':1, 'NBIN_PRD':1,
               'PHS_OFFS':0,
               'NBITS':1,
               'ZERO_OFF':0, 'SIGNINT':0,
               'NSUBOFFS':0,
               'NCHAN':1,
               'CHAN_BW':1,
               'DM':0, 'RM':0, 'NCHNOFFS':0,
               'NSBLK':1}}


Mark5B_header_parsers = OrderedDict()
for k, v in (('sync_pattern', (0, 0, 32)),
             ('year', (1, 28, 4)),
             ('user', (1, 16, 12)),
             ('internal_tvg', (1, 15, 1)),
             ('frame_nr', (1, 0, 15)),
             ('bcd_jday', (2, 20, 12)),
             ('bcd_seconds', (2, 0, 20)),
             ('bcd_fraction', (3, 16, 16)),
             ('crcc', (3, 0, 16))):
    Mark5B_header_parsers[k] = make_parser(*v)

ref_max = 16
ref_epochs = Time(['{y:04d}-01-01'.format(y=2000 + ref)
                   for ref in range(ref_max)], format='isot', scale='utc')


class Mark5BFrameHeader(object):

    _parsers = Mark5B_header_parsers
    size = 16
    payloadsize = 2500 * 4
    framesize = size + payloadsize

    def __init__(self, data, kday=None, verify=True):
        """Interpret a tuple of words as a Mark5B Frame Header."""
        self.data = data
        if kday:
            self.kday = kday
        else:
            self.kday = round(ref_epochs[self['year']].mjd, -3)

        if verify:
            self.verify()

    def verify(self):
        """Verify header integrity."""
        assert len(self.data) == 4
        assert self['sync_pattern'] == SYNC_PATTERN

    @classmethod
    def frombytes(cls, s, *args, **kwargs):
        """Read Mark5B Header from bytes."""
        return cls(four_word_struct.unpack(s), *args, **kwargs)

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        """Read Mark5B Header from file."""
        s = fh.read(16)
        if len(s) != 16:
            raise EOFError
        return cls(four_word_struct.unpack(s), *args, **kwargs)

    def __getitem__(self, item):
        try:
            return self._parsers[item](self.data)
        except KeyError:
            raise KeyError("Mark5B Frame Header does not contain {0}"
                           .format(item))

    def keys(self):
        return self._parsers.keys()

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        name = self.__class__.__name__
        return ("<{0} {1}>".format(name, (",\n  " + len(name) * " ").join(
            ["{0}: {1}".format(k, (hex(self[k])
                                   if k.startswith('bcd') or k.startswith('sy')
                                   else self[k])) for k in self.keys()])))

    @property
    def jday(self):
        return bcd_decode(self['bcd_jday'])

    @property
    def seconds(self):
        return bcd_decode(self['bcd_seconds'])

    @property
    def ns(self):
        ns = bcd_decode(self['bcd_fraction']) * 100000
        # "unround" the nanoseconds
        return 156250 * ((ns+156249) // 156250)

    def time(self):
        """
        Convert year, BCD time code to Time object.

        Uses 'year', which stores the number of years since 2000, and
        the VLBA BCD Time Code in 'bcd_time1', 'bcd_time2'.
        See http://www.haystack.edu/tech/vlbi/mark5/docs/Mark%205B%20users%20manual.pdf
        """
        return Time(self.kday + self.jday,
                    (self.seconds + 1.e-9 * self.ns) / 86400,
                    format='mjd', scale='utc')


def find_frame(fh, maximum=None, forward=True, quickcheck=True):
    """Look for the first occurrence of a frame, from the current position.

    The search is for the MARK5B_SYNC_PATTERN at a given position
    and at PAYLOADSIZE + HEADERSIZE = 10016 bytes ahead.
    """
    b = PAYLOADSIZE + HEADERSIZE
    if maximum is None:
        maximum = 2 * b
    # Loop over chunks to try to find the frame marker.
    file_pos = fh.tell()
    # First check whether we are right at a frame marker (usually true).
    sync = np.array(SYNC_PATTERN, dtype='<u4').view('4u1')
    if quickcheck and np.all(np.fromstring(fh.read(4), dtype=np.uint8) ==
                             sync):
        fh.seek(file_pos)
        return file_pos

    if forward:
        iterate = range(file_pos, file_pos + maximum, b)
    else:
        iterate = range(file_pos - 2 * b, file_pos - 2 * b - maximum, -b)
    for frame in iterate:
        fh.seek(frame)
        data = np.fromstring(fh.read(2 * b), dtype=np.uint8)
        if len(data) < 2 * b:
            break
        data = data.reshape(2, -1)
        match = ((data[:, 0:-3] == sync[0]) &
                 (data[:, 1:-2] == sync[1]) &
                 (data[:, 2:-1] == sync[2]) &
                 (data[:, 3:] == sync[3]))
        match = match[0] & match[1]
        indices = np.nonzero(match)[0]
        if len(indices):
            fh.seek(file_pos)
            return frame + indices[0] + (0 if forward else b)

    fh.seek(file_pos)
    return None


# Some duplication with mark4.py here: lut2bit = mark4.lut2bit1
# Though lut1bit = -mark4.lut1bit, so perhaps not worth combining.
def init_luts():
    """Set up the look-up tables for levels as a function of input byte."""
    lut2level = np.array([-1.0, 1.0], dtype=np.float32)
    lut4level = np.array([-OPTIMAL_2BIT_HIGH, 1.0, -1.0, OPTIMAL_2BIT_HIGH],
                         dtype=np.float32)
    b = np.arange(256)[:, np.newaxis]
    # 1-bit mode
    l = np.arange(8)
    lut1bit = lut2level[(b >> l) & 1]
    # 2-bit mode
    s = np.arange(0, 8, 2)
    lut2bit = lut4level[(b >> s) & 3]
    return lut1bit, lut2bit

lut1bit, lut2bit = init_luts()


def decode_1bit(frame, nvlbichan, channels=None):
    decoded = lut1bit[frame].reshape(-1, nvlbichan)
    return decoded if channels is None else decoded[:, channels]


def decode_2bit(frame, nvlbichan, channels=None):
    decoded = lut2bit[frame].reshape(-1, nvlbichan)
    return decoded if channels is None else decoded[:, channels]


DECODERS = {1: decode_1bit,
            2: decode_2bit}
