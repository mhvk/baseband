# Licensed under the GPLv3 - see LICENSE
import pytest
import numpy as np
from astropy import units as u
from astropy.time import Time

from ... import mark5b
from ...data import SAMPLE_MARK5B as SAMPLE_FILE
from ...vlbi_base.base import HeaderNotFoundError


class TestCorruptSampleCopy:
    def setup_class(cls):
        with open(SAMPLE_FILE, 'rb') as fh:
            cls.sample_bytes = fh.read()
        with mark5b.open(SAMPLE_FILE, 'rs', sample_rate=32*u.MHz,
                         kday=56000, nchan=8, bps=2) as fs:
            cls.frame_nbytes = fs.header0.frame_nbytes
            cls.frame_rate = fs._frame_rate
            cls.start_time = fs.start_time
            cls.stop_time = fs.stop_time
            cls.data = fs.read()
            cls.frame3 = fs._frame

    def expected_bad_frames(self, missing):
        (start_f, start_r), (stop_f, stop_r) = [
            divmod(s, self.frame_nbytes)
            for s in (missing.start, missing.stop-1)]

        if start_r < 5:  # invariants touched?
            start_f -= 1

        return start_f, stop_f+1

    @pytest.mark.parametrize('missing,expected_bad_start,expected_bad_stop', [
        (slice(20032, 20033), 1, 3),  # First byte of header of frame 2.
        (slice(20096, 20100), 2, 3),  # Part of payload of frame 2.
        (slice(12000, 22000), 1, 3),  # Parts of 1-2.
        (slice(30060, 30070), 3, 4),  # Part of header of frame 3.
        (slice(40063, 40064), 3, 4)])  # Last byte of last frame.
    def test_expected_bad_frames(self, missing, expected_bad_start,
                                 expected_bad_stop):
        bad_start, bad_stop = self.expected_bad_frames(missing)
        assert bad_start == expected_bad_start
        assert bad_stop == expected_bad_stop

    @pytest.mark.parametrize('missing', [
        slice(20032, 20033),  # First byte of header of frame 2.
        slice(20096, 20100),  # Part of payload of frame 2.
        slice(12000, 22000),  # Parts of 1-2.
        slice(30060, 30070),  # Part of header of frame 3.
        slice(40063, 40064)])  # Last byte of last frame.
    def test_bad_frames(self, missing, tmpdir):
        corrupted = (self.sample_bytes[:missing.start]
                     + self.sample_bytes[missing.stop:])
        bad_start, bad_stop = self.expected_bad_frames(missing)

        filename = str(tmpdir.join('corrupted.m5b'))
        with open(filename, 'wb') as fw:
            fw.write(corrupted)

        with mark5b.open(filename, 'rb', nchan=8, bps=2) as fr:
            for i in range(bad_start):
                header = fr.find_header(forward=True, maximum=40000,
                                        check=1)
                assert header['frame_nr'] == i
                fr.seek(16, 1)

            if bad_stop < 4:
                header = fr.find_header(forward=True, maximum=40000,
                                        check=1)
                assert header['frame_nr'] == bad_stop
                fr.seek(16, 1)

                for i in range(bad_stop+1, 4):
                    header = fr.find_header(forward=True, maximum=40000,
                                            check=1)
                    assert header['frame_nr'] == i
                    fr.seek(16, 1)
            else:
                with pytest.raises(HeaderNotFoundError):
                    fr.find_header(forward=True, maximum=40000,
                                   check=1)

    # Have to keep frame 0 intact, as well as header of frame 1.
    @pytest.mark.parametrize('missing', [
        slice(20032, 20033),  # First byte of header of frame 2.
        slice(20096, 20100),  # Part of payload of frame 2.
        slice(12000, 22000),  # Parts of 1-2.
        slice(30060, 30070),  # Part of header of frame 3.
        slice(40063, 40064)])  # Last byte of last frame.
    def test_missing_bytes(self, missing, tmpdir):
        corrupted = (self.sample_bytes[:missing.start]
                     + self.sample_bytes[missing.stop:])
        bad_start, bad_stop = self.expected_bad_frames(missing)

        filename = str(tmpdir.join('corrupted.m5b'))
        with open(filename, 'wb') as fw:
            fw.write(corrupted)
            # Add four more frames to ensure that _last_header is OK.
            for i in range(4, 8):
                header = self.frame3.header.copy()
                header.set_time(self.start_time + i/self.frame_rate,
                                frame_rate=self.frame_rate)
                header.update()
                frame = self.frame3.__class__(header, self.frame3.payload,
                                              valid=False)
                frame.tofile(fw)

        # Check that bad frames are found with verify only.
        with mark5b.open(filename, 'rs', nchan=8, bps=2,
                         kday=56000, verify=True) as fv:
            assert not fv.info.readable
            assert not fv.info.checks['continuous']
            assert 'continuous' in fv.info.errors
            expected_msg = 'While reading at {}'.format(
                bad_start * fv.samples_per_frame)
            assert expected_msg in fv.info.errors['continuous']

            # While only warnings are given when it is fixable.
            fv.verify = 'fix'
            assert fv.info.readable
            assert 'fixable' in fv.info.checks['continuous']
            assert 'continuous' in fv.info.warnings
            assert expected_msg in fv.info.warnings['continuous']
            assert 'problem loading frame' in fv.info.warnings['continuous']

        with mark5b.open(filename, 'rs', sample_rate=32*u.MHz,
                         kday=56000, nchan=8, bps=2) as fr:
            assert fr.start_time == self.start_time
            assert abs(fr.stop_time - self.stop_time
                       - 4 / self.frame_rate) < 1.*u.ns
            with pytest.warns(UserWarning,
                              match='problem loading frame'):
                data = fr.read()

        assert data.shape == (40000, 8)

        expected = self.data.copy().reshape(-1, 5000, 8)
        expected[bad_start:bad_stop] = 0.
        expected.shape = -1, 8
        expected = np.concatenate((expected, np.zeros_like(expected)))

        assert np.all(data == expected)


class TestCorruptFile:
    def setup(self):
        time = Time('2010-11-12T13:14:15')
        self.header0 = mark5b.Mark5BHeader.fromvalues(time=time)
        self.data = np.repeat([[-1, 1], [-3, 3]], 10000, axis=0)
        self.nchan = 2
        self.sample_rate = 100 * u.kHz
        self.kwargs = dict(nchan=self.nchan,
                           sample_rate=self.sample_rate,
                           ref_time=time)

    def fake_file(self, tmpdir, nframes=16):
        filename = str(tmpdir.join('fake.mark5b'))
        with mark5b.open(filename, 'ws', header0=self.header0,
                         sample_rate=self.sample_rate,
                         nchan=self.nchan) as fw:
            for _ in range(nframes):
                fw.write(self.data)
        return filename

    def test_fake_file(self, tmpdir):
        fake_file = self.fake_file(tmpdir)
        with mark5b.open(fake_file, 'rs', **self.kwargs) as fr:
            data = fr.read()

        data = data.reshape((-1,) + self.data.shape)
        assert np.all(data.astype(int) == self.data)

    def corrupt_copy(self, filename, missing):
        corrupt_name = filename.replace('.mark5b', '_corrupt.mark5b')
        with open(filename, 'rb') as fr, \
                open(corrupt_name, 'wb') as fw:
            fw.write(fr.read(missing.start))
            fr.seek(missing.stop)
            fw.write(fr.read())
        return corrupt_name

    @pytest.mark.parametrize('frame_nr', [1, 3, 5, slice(7, 10)])
    def test_missing_frames(self, frame_nr, tmpdir):
        if not isinstance(frame_nr, slice):
            frame_nr = slice(frame_nr, frame_nr+1)
        missing = slice(frame_nr.start * self.header0.frame_nbytes,
                        frame_nr.stop * self.header0.frame_nbytes)
        fake_file = self.fake_file(tmpdir)
        corrupt_file = self.corrupt_copy(fake_file, missing)
        with mark5b.open(corrupt_file, 'rs', **self.kwargs) as fr:
            with pytest.warns(UserWarning, match='problem loading'):
                data = fr.read()

        data = data.reshape((-1,) + self.data.shape)
        assert np.all(data[:frame_nr.start].astype(int) == self.data)
        assert np.all(data[frame_nr.stop:].astype(int) == self.data)
        assert np.all(data[frame_nr] == 0.)

    @pytest.mark.parametrize('missing_bytes', [
        slice(0, 10016),  # Remove whole last frame set.
        slice(0, 16),  # Remove header of last frame.
        slice(8, 16),  # Corrupt header of last frame.
        slice(0, 1),  # Corrupt header byte of last frame.
        slice(10, 11),  # Corrupt header byte of last frame.
        slice(15, 16),  # Corrupt header byte of last frame.
        slice(20, 21),  # Corrupt payload byte of last frame.
        slice(10015, 10016),  # last byte of last frame.
    ])
    def test_missing_end(self, missing_bytes, tmpdir):
        # In all these cases, the data read should just be short.
        missing = slice(missing_bytes.start + 15*self.header0.frame_nbytes,
                        missing_bytes.stop + 15*self.header0.frame_nbytes)
        fake_file = self.fake_file(tmpdir)
        corrupt_file = self.corrupt_copy(fake_file, missing)
        # If the sync pattern is still there, or the whole last frame
        # is missing, we'll get all but one.  Otherwise, the one-but-last
        # header will not be considered OK and we loose that frame too.
        if missing_bytes.start > 5 or (missing_bytes.start == 0
                                       and missing_bytes.stop == 10016):
            expected = 15
        else:
            expected = 14

        with mark5b.open(corrupt_file, 'rs', **self.kwargs) as fr:
            assert fr.size == expected * self.data.size
            data = fr.read()

        data = data.reshape((-1,) + self.data.shape)
        assert np.all(data.astype(int) == self.data)

    @pytest.mark.parametrize('missing_bytes,missing_frames', [
        (slice(10016, 20032), slice(1, 2)),  # Remove frame 1.
        (slice(20000, 20501), slice(1, 3)),  # Corrupt frames 1,2.
        (slice(20032, 20048), slice(1, 3)),  # Missing header 2.
    ])
    def test_missing_middle(self, missing_bytes, missing_frames, tmpdir):
        # In all these cases, some data will be missing.
        fake_file = self.fake_file(tmpdir)
        corrupt_file = self.corrupt_copy(fake_file, missing_bytes)
        with mark5b.open(corrupt_file, 'rs', **self.kwargs) as fr:
            assert fr.size == 16 * self.data.size
            with pytest.warns(UserWarning, match='problem loading frame'):
                data = fr.read()

        data = data.reshape((-1,) + self.data.shape)
        expected = np.stack([self.data] * 16)
        expected[missing_frames] = 0.
        assert np.all(data.astype(int) == expected)
