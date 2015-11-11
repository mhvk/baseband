import warnings
import numpy as np
from astropy import units as u
from astropy.utils import lazyproperty
from .utils import get_frame_rate


__all__ = ['u_sample', 'VLBIStreamBase', 'VLBIStreamReaderBase',
           'VLBIStreamWriterBase']

u_sample = u.def_unit('sample', doc='One sample from a data stream')


class VLBIStreamBase(object):
    """VLBI file wrapper, allowing one to read frames."""

    _frame_class = None

    def __init__(self, fh_raw, header0, nchan, bps, thread_ids,
                 samples_per_frame=None, sample_rate=None):
        self.fh_raw = fh_raw
        self.header0 = header0
        self.nchan = nchan
        self.bps = bps
        self.thread_ids = thread_ids
        self.nthread = nchan if thread_ids is None else len(thread_ids)
        if samples_per_frame is None:
            samples_per_frame = header0.payloadsize * 8 // bps // nchan

        if sample_rate is None:
            fh_raw.seek(0)
            self.frames_per_second = get_frame_rate(fh_raw, type(header0))
            fh_raw.seek(self._frame.size)
            sample_rate = (self.frames_per_second *
                           samples_per_frame).to(u.MHz)
        else:
            self.frames_per_second = (sample_rate /
                                      samples_per_frame).to(u.Hz).value

        self.samples_per_frame = samples_per_frame
        self.sample_rate = sample_rate
        self.offset = 0

    # Providing normal File IO properties.
    def readable(self):
        return self.fh_raw.readable()

    def writable(self):
        return self.fh_raw.writable()

    def seekable(self):
        return self.fh_raw.readable()

    def tell(self, offset=None, unit=None):
        """Return offset (in samples or in the given unit)."""
        if offset is None:
            offset = self.offset

        if unit is None:
            return offset

        if unit == 'time':
            return self.header0.time + self.tell(unit=u.s)

        return (offset * u_sample).to(unit, equivalencies=[(u.s, u.Unit(
            self.samples_per_frame * self.frames_per_second * u_sample))])

    def _frame_info(self):
        offset = (self.offset +
                  self.header0['frame_nr'] * self.samples_per_frame)
        full_frame_nr, extra = divmod(offset, self.samples_per_frame)
        dt, frame_nr = divmod(full_frame_nr, self.frames_per_second)
        return int(dt), int(frame_nr), extra

    def close(self):
        return self.fh_raw.close()

    @property
    def closed(self):
        return self.fh_raw.closed

    @property
    def name(self):
        return self.fh_raw.name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self):
        return ("<{s.__class__.__name__} name={s.name} offset={s.offset}\n"
                "    nchan={s.nchan}, thread_ids={s.thread_ids}, "
                "samples_per_frame={s.samples_per_frame}, bps={s.bps}\n"
                "    sample_rate={s.sample_rate}, (start) time={h.time.isot}>"
                .format(s=self, h=self.header0))


class VLBIStreamReaderBase(VLBIStreamBase):
    _find_frame = None

    @lazyproperty
    def header1(self):
        raw_offset = self.fh_raw.tell()
        self.fh_raw.seek(-self.header0.framesize, 2)
        header1 = self.fh_raw.find_frame(template_header=self.header0,
                                         maximum=10*self.header0.framesize,
                                         forward=False)
        self.fh_raw.seek(raw_offset)
        if header1 is None:
            raise TypeError("Corrupt VLBI frame? No frame in last {0} bytes."
                            .format(10*self.header0.framesize))
        return header1

    @property
    def size(self):
        n_frames = round(
            (self.header1.time - self.header0.time).to(u.s).value *
            self.frames_per_second) + 1
        return n_frames * self.samples_per_frame

    def seek(self, offset, from_what=0):
        """Like normal seek, but with the offset in samples."""
        if from_what == 0:
            self.offset = offset
        elif from_what == 1:
            self.offset += offset
        elif from_what == 2:
            self.offset = self.size + offset
        return self.offset


class VLBIStreamWriterBase(VLBIStreamBase):
    def close(self):
        extra = self.offset % self.samples_per_frame
        if extra != 0:
            warnings.warn("Closing with partial buffer remaining."
                          "Writing padded frame, marked as invalid.")
            self.write(np.zeros((extra, self.nthread, self.nchan)),
                       invalid_data=True)
            assert self.offset % self.samples_per_frame == 0
        return super(VLBIStreamWriterBase, self).close()
