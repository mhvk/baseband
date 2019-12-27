# Licensed under the GPLv3 - see LICENSE
import pytest
import numpy as np
from astropy import units as u
from astropy.time import Time

from ... import vdif
from ...data import SAMPLE_VDIF as SAMPLE_FILE


class TestCorruptSampleCopy:
    def setup(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            self.sample_bytes = fh.read()
        with vdif.open(SAMPLE_FILE, 'rs') as fs:
            self.frame_nbytes = fs.header0.frame_nbytes
            self.data = fs.read()
            self.start_time = fs.start_time
            self.stop_time = fs.stop_time

        self.thread_ids = [1, 3, 5, 7, 0, 2, 4, 6]
        data = []
        with vdif.open(SAMPLE_FILE, 'rb') as fr:
            # Also create data in order of reading it from disk.
            for i in range(0, fs.size, fs.samples_per_frame):
                data.append(fr.read_frame().data)

        self.disk_ordered = np.concatenate(data)
        self.reverse_threads = [self.thread_ids.index(thread_id)
                                for thread_id in range(8)]

    def expected_bad_frames(self, missing):
        (start_f, start_r), (stop_f, stop_i) = [
            divmod(s, self.frame_nbytes)
            for s in (missing.start, missing.stop-1)]

        if start_r < 32 and start_f % 8 != 0:
            start_f -= 1

        return start_f, stop_f+1

    @pytest.mark.parametrize('missing,expected_bad_start,expected_bad_stop', [
        (slice(50320, 50321), 9, 11),  # First byte of header of frame 10.
        (slice(50500, 50600), 10, 11),  # Part of payload of frame 10.
        (slice(60000, 70000), 11, 14),  # Parts of 11-13.
        (slice(75490, 75500), 14, 16),  # Part of header of frame 15.
        (slice(80511, 80512), 15, 16)])  # Last byte of last frame.
    def test_expected_bad_frames(self, missing, expected_bad_start,
                                 expected_bad_stop):
        bad_start, bad_stop = self.expected_bad_frames(missing)
        assert bad_start == expected_bad_start
        assert bad_stop == expected_bad_stop

    # Have to keep first frameset intact, as well as the
    # first frame of the seconds one.
    @pytest.mark.parametrize('missing', [
        (slice(50320, 50321)),  # First byte of header of frame 10.
        (slice(50500, 50600)),  # Part of payload of frame 10.
        (slice(60000, 70000)),  # Parts of 11-13.
        (slice(75490, 75500)),  # Part of header of frame 15.
        (slice(80511, 80512))])  # Last byte of last frame.
    def test_missing_bytes(self, missing, tmpdir):
        corrupted = (self.sample_bytes[:missing.start]
                     + self.sample_bytes[missing.stop:])
        bad_start, bad_stop = self.expected_bad_frames(missing)

        filename = str(tmpdir.join('corrupted.vdif'))
        with open(filename, 'wb') as fw:
            fw.write(corrupted)

        with vdif.open(filename, 'rs') as fr:
            assert fr.start_time == self.start_time
            assert fr.stop_time == self.stop_time
            with pytest.warns(UserWarning,
                              match='problem loading frame'):
                data = fr.read()

        expected = self.disk_ordered.copy().reshape(-1, 20000)
        expected[bad_start:bad_stop] = 0.

        # Mimic thread ordering
        expected = (expected.reshape(-1, 8, 20000)
                    .transpose(0, 2, 1).reshape(-1, 8)
                    [:, self.reverse_threads])
        assert np.all(data == expected)


class TestCorruptFile:
    def setup(self):
        self.header0 = vdif.VDIFHeader.fromvalues(
            edv=1, time=Time('2010-11-12T13:14:15'), nchan=2, bps=2,
            complex_data=False, thread_id=0, samples_per_frame=16,
            station='me', sample_rate=2*u.kHz)
        self.nthread = 2
        self.data = np.array([[[-1, 1],
                               [-3, 3]]]*16)
        self.frameset_nbytes = self.header0.frame_nbytes * self.nthread

    def fake_file(self, tmpdir, nframes=16):
        filename = str(tmpdir.join('fake.vdif'))
        with vdif.open(filename, 'ws', header0=self.header0,
                       nthread=self.nthread) as fw:
            for _ in range(nframes):
                fw.write(self.data)
        return filename

    def corrupt_copy(self, filename, missing):
        corrupt_name = filename.replace('.vdif', '_corrupt.vdif')
        with open(filename, 'rb') as fr, \
                open(corrupt_name, 'wb') as fw:
            fw.write(fr.read(missing.start))
            fr.seek(missing.stop)
            fw.write(fr.read())
        return corrupt_name

    @pytest.mark.parametrize('frame_nr', [1, 3, 5, slice(7, 10)])
    def test_missing_frameset(self, frame_nr, tmpdir):
        if not isinstance(frame_nr, slice):
            frame_nr = slice(frame_nr, frame_nr+1)
        missing = slice(frame_nr.start * self.frameset_nbytes,
                        frame_nr.stop * self.frameset_nbytes)
        fake_file = self.fake_file(tmpdir)
        corrupt_file = self.corrupt_copy(fake_file, missing)
        with vdif.open(corrupt_file, 'rs') as fr:
            with pytest.warns(UserWarning, match='All threads'):
                data = fr.read()

        data = data.reshape((-1,) + self.data.shape)
        assert np.all(data[:frame_nr.start].astype(int) == self.data)
        assert np.all(data[frame_nr.stop:].astype(int) == self.data)
        assert np.all(data[frame_nr] == 0.)

    @pytest.mark.parametrize('frame_nr,thread', [
        (3, 0), (3, 1), (1, 1), (15, 1)])
    def test_missing_thread(self, frame_nr, thread, tmpdir):
        frame = frame_nr * self.nthread + thread
        missing = slice(frame * self.header0.frame_nbytes,
                        (frame+1) * self.header0.frame_nbytes)
        fake_file = self.fake_file(tmpdir)
        corrupt_file = self.corrupt_copy(fake_file, missing)
        with vdif.open(corrupt_file, 'rs') as fr:
            with pytest.warns(UserWarning,
                              match='Thread.*{0}.*missing'.format(thread)):
                data = fr.read()

        data = data.reshape((-1,) + self.data.shape)
        assert np.all(data[:frame_nr].astype(int) == self.data)
        assert np.all(data[frame_nr+1:].astype(int) == self.data)
        assert np.all(data[frame_nr, :, thread] == 0.)
        assert np.all(data[frame_nr, :, 1-thread].astype(int)
                      == self.data[:, 1-thread])

    @pytest.mark.parametrize('missing_bytes', [
        slice(0, 80),  # Remove whole last frame set.
        slice(0, 40),  # Remove first thread of last frame
        slice(0, 32),  # Remove first header of last frame.
        slice(16, 32),  # Corrupt first header of last frame.
        slice(0, 16),  # Corrupt first header of last frame.
        slice(0, 1),  # Corrupt header byte of last frame.
        slice(10, 11),  # Corrupt header byte of last frame.
        slice(15, 16),  # Corrupt header byte of last frame.
        slice(20, 21),  # Corrupt header byte of last frame.
        slice(23, 24),  # Corrupt header byte of last frame.
    ])
    def test_missing_end(self, missing_bytes, tmpdir):
        # In all these cases, the data read should just be short.
        missing = slice(missing_bytes.start + 15*self.frameset_nbytes,
                        missing_bytes.stop + 15*self.frameset_nbytes)
        fake_file = self.fake_file(tmpdir)
        corrupt_file = self.corrupt_copy(fake_file, missing)
        with vdif.open(corrupt_file, 'rs') as fr:
            assert fr.size == 15 * self.data.size
            data = fr.read()

        data = data.reshape((-1,) + self.data.shape)
        assert len(data) == 15
        assert np.all(data.astype(int) == self.data)

    @pytest.mark.parametrize('missing_bytes,missing_data', [
        (slice(80, 160), slice(16, 32)),  # Remove frame set 1.
        (slice(119, 121), slice(16, 32)),  # Corrupt frame set 1.
        (slice(120, 121), slice(16, 32)),  # Corrupt frame 1, thread 1 header.
        (slice(119, 120), slice(16, 32)),  # Corrupt frame 1, thread 0.
        (slice(112, 205), slice(16, 48)),  # Corrupt frames 1, 2.
    ])
    def test_missing_middle(self, missing_bytes, missing_data, tmpdir):
        # In all these cases, some data will be missing.
        fake_file = self.fake_file(tmpdir)
        corrupt_file = self.corrupt_copy(fake_file, missing_bytes)
        with vdif.open(corrupt_file, 'rs') as fr:
            assert fr.size == 16 * self.data.size
            with pytest.warns(UserWarning, match='problem loading frame set'):
                data = fr.read()

        expected = np.concatenate([self.data] * 16)
        expected[missing_data] = 0.
        assert np.all(data.astype(int) == expected)
