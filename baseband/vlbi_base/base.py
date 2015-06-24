from astropy import units as u
from .utils import get_frame_rate

u_sample = u.def_unit('sample')


class VLBIStreamBase(object):
    """VLBI file wrapper, allowing one to read frames."""

    _frame_class = None

    def __init__(self, fh_raw, header0, nchan, bps, thread_ids,
                 sample_rate=None):
        self.fh_raw = fh_raw
        self.header0 = header0
        self.nchan = nchan
        self.bps = bps
        self.thread_ids = thread_ids
        self.nthread = nchan if thread_ids is None else len(thread_ids)
        self.samples_per_frame = header0.payloadsize * 8 // bps // nchan
        if sample_rate is None:
            fh_raw.seek(0)
            self.frames_per_second = get_frame_rate(fh_raw, type(header0))
            fh_raw.seek(self._frame.size)
            sample_rate = (self.frames_per_second *
                           self.samples_per_frame).to(u.MHz)
        else:
            self.frames_per_second = (sample_rate /
                                      self.samples_per_frame).to(u.Hz).value
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

        if unit == 'frame_info':
            full_frame_nr, extra = divmod(offset, self.samples_per_frame)
            dt, frame_nr = divmod(full_frame_nr, self.frames_per_second)
            return dt, frame_nr, extra

        if unit == 'time':
            return self.header0.time() + self.tell(u.s)

        return (offset * u_sample).to(
            unit, equivalencies=[(self.samples_per_frame * u.sample,
                                  self.frames_per_second * u.Hz)])

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
