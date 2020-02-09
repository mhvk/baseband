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

    def test_sample_bytes(self, tmpdir):
        test_file = str(tmpdir.join('test.vdif'))
        with open(test_file, 'wb') as fh:
            fh.write(self.sample_bytes)
        with vdif.open(test_file, 'rs') as fs:
            data = fs.read()
        assert np.all(data == self.data)

    # Have 6 framesets, so 48 frames.
    @pytest.mark.parametrize('missing', (
        36, slice(46, 48), [30, 45], slice(8, 16), 0, slice(4, 12)))
    def test_missing_frames(self, missing, tmpdir):
        """Purely missing frames should just be marked invalid."""
        # Even at the very start; gh-359
        sample = np.frombuffer(self.sample_bytes, 'u1').reshape(-1, 5032)
        use = np.ones(len(sample), bool)
        use[missing] = False
        reduced = sample[use]
        corrupt_file = str(tmpdir.join('missing_frames.vdif'))
        with open(corrupt_file, 'wb') as s:
            s.write(reduced.tostring())

        with vdif.open(corrupt_file, 'rb') as fr:
            assert 'number_of_frames' not in fr.info.warnings
            if np.count_nonzero(use) % 8 == 0:
                assert 'number_of_framesset' not in fr.info.warnings
            else:
                assert 'number_of_framesets' in fr.info.warnings

        with vdif.open(corrupt_file, 'rs') as fh:
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

        with vdif.open(filename, 'rb') as fr:
            assert 'number_of_frames' in fr.info.warnings

        # Check that bad frames are found with verify only.
        with vdif.open(filename, 'rs', verify=True) as fv:
            assert not fv.info.readable
            assert not fv.info.checks['continuous']
            assert 'continuous' in fv.info.errors
            # Reading will fail the frameset *before* the one tested.
            expected_msg = 'While reading at {}'.format(
                (bad_start // 8 - 1) * fv.samples_per_frame)
            assert expected_msg in fv.info.errors['continuous']

        # While only warnings are given when it is fixable.
        with vdif.open(filename, 'rs', verify='fix') as ff:
            assert ff.info.readable
            assert 'fixable' in ff.info.checks['continuous']
            assert 'continuous' in ff.info.warnings
            assert expected_msg in ff.info.warnings['continuous']
            assert 'problem loading frame' in ff.info.warnings['continuous']

        # Now check that data is properly marked as invalid.
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


class TestInvalidFrameHeaders:
    # CHIME VDIF files from ARO can have invalid frames included in
    # which the header information -- in particular the frame_nr and
    # seconds -- are corrupt as well.  Check that we can skip those.
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

    def fake_file(self, tmpdir):
        filename = str(tmpdir.join('fake.vdif'))
        with vdif.open(filename, 'wb') as fw:
            for i in range(16):
                header = self.header0.copy()
                if i != 10:
                    header['frame_nr'] = i
                else:
                    header['frame_nr'] = 0
                    header['seconds'] = 0
                    header['invalid_data'] = True
                fw.write_frameset(self.data, header=header)

        return filename

    @pytest.mark.parametrize('verify', ('fix', False))
    def test_invalid_frame_fix(self, verify, tmpdir):
        fake_file = self.fake_file(tmpdir)
        with vdif.open(fake_file, 'rs', verify=verify) as fr:
            data = fr.read()

        expected = np.stack([self.data] * 16)
        expected[10] = 0
        expected.shape = (-1,) + self.data.shape[1:]
        assert np.all(data.astype(int) == expected)

    def test_invalid_frame_nofix(self, tmpdir):
        fake_file = self.fake_file(tmpdir)
        with pytest.raises(ValueError, match='wrong frame number'):
            with vdif.open(fake_file, 'rs', verify=True) as fr:
                fr.read()
