# Licensed under the GPLv3 - see LICENSE
import pytest
import numpy as np
from astropy import units as u
from astropy.time import Time

from ... import mark4


class TestCorruptFile:
    def setup_class(cls):
        time = Time('2010-11-12T13:14:15')
        cls.nchan = 2
        cls.header0 = mark4.Mark4Header.fromvalues(
            time=time, ntrack=16, nchan=cls.nchan, fanout=4)
        cls.data = np.zeros((2*cls.header0.frame_nbytes, 2))
        cls.data.reshape(-1, 4, 2)[160:] = [
            [-1, 1], [-3, 3], [1, -1], [3, -3]]
        cls.sample_rate = 100 * u.kHz
        cls.kwargs = dict(sample_rate=cls.sample_rate,
                          ref_time=time)

    def fake_file(self, tmpdir, nframes=8):
        filename = str(tmpdir.join('fake.mark4'))
        with mark4.open(filename, 'ws', header0=self.header0,
                        sample_rate=self.sample_rate) as fw:
            for _ in range(nframes):
                fw.write(self.data)
        return filename

    def test_fake_file(self, tmpdir):
        fake_file = self.fake_file(tmpdir)
        with mark4.open(fake_file, 'rs', **self.kwargs) as fr:
            data = fr.read()

        data = data.reshape((-1,) + self.data.shape)
        assert np.all(data.astype(int) == self.data)

    def corrupt_copy(self, filename, missing):
        corrupt_name = filename.replace('.mark4', '_corrupt.mark4')
        with open(filename, 'rb') as fr, \
                open(corrupt_name, 'wb') as fw:
            fw.write(fr.read(missing.start))
            fr.seek(missing.stop)
            fw.write(fr.read())
        return corrupt_name

    @pytest.mark.parametrize('frame_nr', [1, 3, slice(3, 5)])
    def test_missing_frames(self, frame_nr, tmpdir):
        if not isinstance(frame_nr, slice):
            frame_nr = slice(frame_nr, frame_nr+1)
        missing = slice(frame_nr.start * self.header0.frame_nbytes,
                        frame_nr.stop * self.header0.frame_nbytes)
        fake_file = self.fake_file(tmpdir)
        corrupt_file = self.corrupt_copy(fake_file, missing)

        # Check that bad frames are found with verify only.
        with mark4.open(corrupt_file, 'rs', verify=True, **self.kwargs) as fv:
            assert not fv.info.readable
            assert not fv.info.checks['continuous']
            assert 'continuous' in fv.info.errors
            expected_msg = 'While reading at {}'.format(
                frame_nr.start * fv.samples_per_frame)
            assert expected_msg in fv.info.errors['continuous']

        # While only warnings are given when it is fixable.
        with mark4.open(corrupt_file, 'rs', verify='fix', **self.kwargs) as ff:
            assert ff.info.readable
            assert 'fixable' in ff.info.checks['continuous']
            assert 'continuous' in ff.info.warnings
            assert expected_msg in ff.info.warnings['continuous']
            assert 'problem loading frame' in ff.info.warnings['continuous']

        with mark4.open(corrupt_file, 'rs', **self.kwargs) as fr:
            with pytest.warns(UserWarning, match='problem loading'):
                data = fr.read()

        data = data.reshape((-1,) + self.data.shape)
        assert np.all(data[:frame_nr.start].astype(int) == self.data)
        assert np.all(data[frame_nr.stop:].astype(int) == self.data)
        assert np.all(data[frame_nr] == 0.)

    @pytest.mark.parametrize('missing_bytes', [
        slice(0, 40000),  # Remove whole last frame set.
        slice(0, 320),  # Remove header of last frame.
        slice(8, 16),  # Corrupt header of last frame.
        slice(0, 1),  # Corrupt header byte of last frame.
        slice(10, 11),  # Corrupt header byte of last frame.
        slice(319, 320),  # Corrupt header byte of last frame.
        slice(400, 401),  # Corrupt payload byte of last frame.
        slice(39999, 40000),  # last byte of last frame.
    ])
    def test_missing_end(self, missing_bytes, tmpdir):
        # In all these cases, the data read should just be short, but
        # it varies by how much: If the sync pattern is still there,
        # or the whole last frame is missing, we'll lose the last
        # frame.  Otherwise, the one-but-last header will not be
        # considered OK and we lose that frame too.
        if missing_bytes.start > 192 or (missing_bytes.start == 0
                                         and missing_bytes.stop == 40000):
            expected = 7
        else:
            expected = 6
        missing = slice(missing_bytes.start + 7*self.header0.frame_nbytes,
                        missing_bytes.stop + 7*self.header0.frame_nbytes)
        fake_file = self.fake_file(tmpdir)
        corrupt_file = self.corrupt_copy(fake_file, missing)
        with mark4.open(corrupt_file, 'rs', **self.kwargs) as fr:
            assert len(fr.fh_raw.locate_frames(
                fr.header0, maximum=1000000)) == expected
            assert fr.size == expected * self.data.size
            data = fr.read()

        data = data.reshape((-1,) + self.data.shape)
        assert np.all(data.astype(int) == self.data)

    @pytest.mark.parametrize('missing_bytes,missing_frames', [
        (slice(40000, 80000), slice(1, 2)),  # Remove frame 1.
        (slice(78000, 82000), slice(1, 3)),  # Corrupt frames 1,2.
        (slice(80010, 80100), slice(1, 3)),  # Corrupted header 2.
    ])
    def test_missing_middle(self, missing_bytes, missing_frames, tmpdir):
        # In all these cases, some data will be missing.
        fake_file = self.fake_file(tmpdir)
        corrupt_file = self.corrupt_copy(fake_file, missing_bytes)
        # Check that bad frames are found with verify only.
        with mark4.open(corrupt_file, 'rs', verify=True, **self.kwargs) as fv:
            assert not fv.info.readable
            assert not fv.info.checks['continuous']
            assert 'continuous' in fv.info.errors
            expected_msg = 'While reading at {}'.format(
                missing_frames.start * fv.samples_per_frame)
            assert expected_msg in fv.info.errors['continuous']

        with mark4.open(corrupt_file, 'rs', **self.kwargs) as fr:
            assert fr.size == 8 * self.data.size
            with pytest.warns(UserWarning, match='problem loading frame'):
                data = fr.read()

        data = data.reshape((-1,) + self.data.shape)
        expected = np.stack([self.data] * 8)
        expected[missing_frames] = 0.
        assert np.all(data.astype(int) == expected)
