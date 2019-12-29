# Licensed under the GPLv3 - see LICENSE
import io

import pytest
import numpy as np
from astropy import units as u
from astropy.time import Time

from ... import vdif
from ...data import SAMPLE_VDIF as SAMPLE_FILE


class TestCorruptSampleCopy:
    @classmethod
    def setup_class(cls):
        # Make a triply-long sample file - this since otherwise
        # things already fail at the determination of thread_ids.
        with vdif.open(SAMPLE_FILE, 'rs') as fs, \
                io.BytesIO() as s, \
                vdif.open(s, 'ws', header0=fs.header0, nthread=8) as fw:

            data = fs.read()
            for i in range(3):
                fw.write(data)

            cls.data = np.concatenate([data, data, data])

            s.seek(0)
            cls.sample_bytes = s.read()
            cls.frame_nbytes = fw.header0.frame_nbytes
            cls.start_time = fw.start_time
            cls.stop_time = fw.tell('time')

    def test_sample_bytes(self):
        with io.BytesIO(self.sample_bytes) as s, \
                vdif.open(s, 'rs') as fs:
            data = fs.read()
        assert np.all(data == self.data)

    # Have 6 framesets, so 48 frames.
    @pytest.mark.parametrize('missing', (
        36, slice(46, 48), [30, 45], slice(8, 16), 0, slice(4, 12)))
    def test_missing_frames(self, missing):
        """Purely missing frames should just be marked invalid."""
        # Even at the very start; gh-359
        sample = np.frombuffer(self.sample_bytes, 'u1').reshape(-1, 5032)
        use = np.ones(len(sample), bool)
        use[missing] = False
        reduced = sample[use]
        with io.BytesIO() as s:
            s.write(reduced.tostring())
            s.seek(0)
            with vdif.open(s, 'rs') as fh:
                with pytest.warns(UserWarning,
                                  match='problem loading frame'):
                    data = fh.read()

        # Get data in frame order to zero expected bad frames.
        expected = (self.data.copy().reshape(-1, 20000, 8)
                    .transpose(0, 2, 1).reshape(-1, 20000))
        expected[missing] = 0.
        # Back into regular order
        expected = (expected.reshape(-1, 8, 20000)
                    .transpose(0, 2, 1).reshape(-1, 8))

        assert np.all(expected == data)

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

    # Keep frames in first three frame sets intact for get_thread_ids()
    @pytest.mark.parametrize('missing', [
        (slice(5032*26, 5032*26+1)),  # First byte of header of frame 26.
        (slice(5032*26+50, 5032*26+60)),  # Part of payload of frame 26.
        (slice(5032*27+50, 5032*29+700)),  # Parts of 27-29
        (slice(5032*31+10, 5032*31+20)),  # Part of header of frame 31.
        (slice(5032*48-1, 5032*48))])  # Last byte of last frame.
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

        # Get data in frame order to zero expected bad frames.
        expected = (self.data.copy().reshape(-1, 20000, 8)
                    .transpose(0, 2, 1).reshape(-1, 20000))
        expected[bad_start:bad_stop] = 0.
        # Back into regular order
        expected = (expected.reshape(-1, 8, 20000)
                    .transpose(0, 2, 1).reshape(-1, 8))

        assert np.all(data == expected)


class TestCorruptFile:
    @classmethod
    def setup_class(cls):
        cls.header0 = vdif.VDIFHeader.fromvalues(
            edv=1, time=Time('2010-11-12T13:14:15'), nchan=2, bps=2,
            complex_data=False, thread_id=0, samples_per_frame=16,
            station='me', sample_rate=2*u.kHz)
        cls.nthread = 2
        cls.data = np.array([[[-1, 1],
                              [-3, 3]]]*16)
        cls.frameset_nbytes = cls.header0.frame_nbytes * cls.nthread

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

    # Note: keep frame sets 0--2 intact for get_thread_ids().
    @pytest.mark.parametrize('missing_bytes,missing_data', [
        (slice(240, 320), slice(48, 64)),  # Remove frameset 3.
        (slice(279, 281), slice(48, 64)),  # Corrupt frameset 3.
        (slice(280, 281), slice(48, 64)),  # Corrupt frameset 3, thread 1.
        (slice(279, 280), slice(48, 64)),  # Corrupt frameset 3, thread 0.
        (slice(272, 365), slice(48, 80)),  # Corrupt framesets 3, 4
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
