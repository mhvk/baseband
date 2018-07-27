from ..vlbi_base.base import VLBIStreamReaderBase
from .frame import ASPFrame
from .header import ASPHeader
import astropy.units as u

class ASPStreamReader(VLBIStreamReaderBase):

    # unfinished
    def __init__(self, fh_raw):
        self.fh_raw = fh_raw
        pos = fh_raw.tell()
        header0 = ASPHeader.fromfile(fh_raw)
        fh_raw.seek(pos)
        self._header0 = header0
        sample_rate = header0['ch_bw'][0] * u.MHz
        samples_per_frame = header0['NPtsSend'][0]
        super(ASPStreamReader, self).__init__(fh_raw, header0, sample_rate,
            samples_per_frame, unsliced_shape = None, bps=8, complex_data=True,
            squeeze = False, subset = None, fill_value=0.0, verify=False)

    def read_frame(self):
        return ASPFrame.fromfile(self.fh_raw)