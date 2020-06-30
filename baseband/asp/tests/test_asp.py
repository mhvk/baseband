from ... import asp
from ...data import SAMPLE_ASP as SAMPLE_FILE


__all__ = ['TestASP']


class TestASP:
    def setup(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            sr = asp.ASPStreamReader(fh)
            self.frame = sr.read_frame()
            self.header = self.frame.header
            self.payload = self.frame.payload
            assert fh.tell() == (
                self.header.nbytes
                + self.header.file_header.nbytes
                + self.payload.nbytes), 'not at end of payload'

    def test_header(self, tmpdir=None):
        """For now this is a very modest test of read integrity for the asp format
        """
        if self.frame is None:
            self.setup()
        header = self.header
        assert header.nbytes == 44
        assert header['totalsize'] == 512
        assert header['nptssend'] == 128
        assert header['freqchanno'] == 10
