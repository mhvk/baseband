# Licensed under the GPLv3 - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
import copy
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.tests.helper import catch_warnings
from ... import dada
from ..base import DADAFileNameSequencer
from ...data import SAMPLE_DADA as SAMPLE_FILE


class TestDADA(object):
    def setup(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            self.header = dada.DADAHeader.fromfile(fh)
            self.payload = dada.DADAPayload.fromfile(fh, self.header)

    def test_header(self, tmpdir):
        with open(SAMPLE_FILE, 'rb') as fh:
            header = dada.DADAHeader.fromfile(fh)
            assert header.size == 4096
            assert fh.tell() == 4096
        assert header['NDIM'] == 2
        assert header['NCHAN'] == 1
        assert header['UTC_START'] == '2013-07-02-01:37:40'
        assert header['OBS_OFFSET'] == 6400000000  # 100 s
        assert header.time.isot == '2013-07-02T01:39:20.000'
        assert header.framesize == 64000 + 4096
        assert header.payloadsize == 64000
        assert header.mutable is False
        with pytest.raises(TypeError):
            header['NCHAN'] = 2
        assert header['NCHAN'] == 1
        # access key via attribute
        assert header.nchan == 1
        with pytest.raises(AttributeError):
            header.python

        with open(str(tmpdir.join('test.dada')), 'w+b') as s:
            header.tofile(s)
            assert s.tell() == header.size
            s.seek(0)
            header2 = dada.DADAHeader.fromfile(s)
            assert header2 == header
            assert header2.mutable is False
            assert s.tell() == header.size

        with open(str(tmpdir.join('test.dada')), 'w+b') as s:
            # now create header with wrong HDR_SIZE in file
            bad_header = header.copy()
            bad_header['HDR_SIZE'] = 1000
            with pytest.raises(ValueError):
                bad_header.tofile(s)
            # now write header explicitly, skipping the check in tofile,
            # so we create a bad header.
            for line in bad_header._tolines():
                s.write((line + '\n').encode('ascii'))
            s.write('# end of header\n'.encode('ascii'))
            s.seek(0)
            with catch_warnings(UserWarning) as w:
                dada.DADAHeader.fromfile(s)
            assert 'Odd' in str(w[0].message)

        # Note that this is not guaranteed to preserve order!
        header3 = dada.DADAHeader.fromkeys(**header)
        assert header3 == header
        assert header3.mutable is True
        # check attribute setting.
        header3.start_time = header.start_time - 0.5 * u.day
        assert np.abs(header3.start_time -
                      (header.start_time - 0.5 * u.day)) < 1 * u.ns
        assert np.abs(header3.time - (header.time - 0.5 * u.day)) < 1 * u.ns
        # check against rounding.
        just_below_int = Time(55000, -1e-15, format='mjd')
        header3.start_time = just_below_int
        assert header3['MJD_START'] == '54999.999999999999999'
        header3['NCHAN'] = 2
        assert header3['NCHAN'] == 2
        header3.framesize = 9096
        assert header3.payloadsize == 5000
        # # Try initialising with properties instead of keywords.
        # Here, we first just try the start time.
        header4 = dada.DADAHeader.fromvalues(
            start_time=header.start_time,
            offset=header.time-header.start_time,
            bps=header.bps, complex_data=header.complex_data,
            sample_rate=header.sample_rate, sideband=header.sideband,
            samples_per_frame=header.samples_per_frame,
            sample_shape=header.sample_shape,
            source=header['SOURCE'], ra=header['RA'], dec=header['DEC'],
            telescope=header['TELESCOPE'], instrument=header['INSTRUMENT'],
            receiver=header['RECEIVER'], freq=header['FREQ'],
            pic_version=header['PIC_VERSION'])
        assert header4 == header
        assert header4.mutable is True
        # And now try both start time and time of observation.
        header5 = dada.DADAHeader.fromvalues(
            offset=header.offset, time=header.time,
            bps=header.bps, complex_data=header.complex_data,
            sample_rate=header.sample_rate, sideband=header.sideband,
            samples_per_frame=header.samples_per_frame,
            sample_shape=header.sample_shape,
            source=header['SOURCE'], ra=header['RA'], dec=header['DEC'],
            telescope=header['TELESCOPE'], instrument=header['INSTRUMENT'],
            receiver=header['RECEIVER'], freq=header['FREQ'],
            pic_version=header['PIC_VERSION'])
        assert header5 == header
        # Check repr can be used to instantiate header
        header6 = eval('dada.' + repr(header))
        assert header6 == header
        # repr includes the comments
        assert header6.comments == header.comments
        # Therefore repr should be identical too.
        assert repr(header6) == repr(header)
        # Check instantiation via tuple
        header7 = dada.DADAHeader(((key, (header[key], header.comments[key]))
                                   for key in header))
        assert header7 == header
        assert header7.comments == header.comments
        # Check copying
        header8 = header.copy()
        assert header8 == header
        assert header8.mutable is True
        assert header8.comments == header.comments
        header9 = copy.copy(header8)
        assert header9 == header
        assert header9.mutable is True
        assert header9.comments == header.comments

    def test_payload(self, tmpdir):
        payload = self.payload
        assert payload.size == 64000
        assert payload.shape == (16000, 2, 1)
        # Check sample shape validity
        assert payload.sample_shape == (2, 1)
        assert payload.sample_shape.npol == 2
        assert payload.sample_shape.nchan == 1
        assert payload.dtype == np.complex64
        assert np.all(payload[:3] == np.array(
            [[[-38.-38.j], [-38.-38.j]],
             [[-38.-38.j], [-40.+0.j]],
             [[-105.+60.j], [85.-15.j]]], dtype=np.complex64))

        with open(str(tmpdir.join('test.dada')), 'w+b') as s:
            payload.tofile(s)
            s.seek(0)
            payload2 = dada.DADAPayload.fromfile(s, payloadsize=64000, bps=8,
                                                 complex_data=True,
                                                 sample_shape=(2, 1))
            assert payload2 == payload
            with pytest.raises(EOFError):
                # Too few bytes.
                s.seek(100)
                dada.DADAPayload.fromfile(s, self.header)
        payload3 = dada.DADAPayload.fromdata(payload.data, bps=8)
        assert payload3 == payload
        with open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(4096)
            payload4 = dada.DADAPayload.fromfile(fh, self.header, memmap=True)
        assert isinstance(payload4.words, np.memmap)
        assert not isinstance(payload.words, np.memmap)
        assert payload == payload4

    def test_frame(self, tmpdir):
        with dada.open(SAMPLE_FILE, 'rb') as fh:
            frame = fh.read_frame(memmap=False)
        header, payload = frame.header, frame.payload
        assert header == self.header
        assert payload == self.payload
        assert frame == dada.DADAFrame(header, payload)
        assert np.all(frame[:3] == np.array(
            [[[-38.-38.j], [-38.-38.j]],
             [[-38.-38.j], [-40.+0.j]],
             [[-105.+60.j], [85.-15.j]]], dtype=np.complex64))
        with open(str(tmpdir.join('test.dada')), 'w+b') as s:
            frame.tofile(s)
            s.seek(0)
            frame2 = dada.DADAFrame.fromfile(s, memmap=False)
        assert frame2 == frame
        frame3 = dada.DADAFrame.fromdata(payload.data, header)
        assert frame3 == frame
        frame4 = dada.DADAFrame.fromdata(payload.data, **header)
        assert frame4 == frame
        header5 = header.copy()
        frame5 = dada.DADAFrame(header5, payload, valid=False)
        assert frame5.valid is False
        assert np.all(frame5.data == 0.)
        frame5.valid = True
        assert frame5 == frame

    def test_frame_memmap(self, tmpdir):
        # Get frame regular way.
        with dada.open(SAMPLE_FILE, 'rb') as fr:
            frame = fr.read_frame(memmap=False)
        assert not isinstance(frame.payload.words, np.memmap)
        # Check that if we map it instead, we get the same result.
        with dada.open(SAMPLE_FILE, 'rb') as fh:
            frame2 = fh.read_frame(memmap=True)
        assert frame2 == frame
        assert isinstance(frame2.payload.words, np.memmap)
        # Bit superfluous perhaps, but check decoding as well.
        assert np.all(frame2[:3] == np.array(
            [[[-38.-38.j], [-38.-38.j]],
             [[-38.-38.j], [-40.+0.j]],
             [[-105.+60.j], [85.-15.j]]], dtype=np.complex64))
        assert np.all(frame2.data == frame.data)

        # Now check writing.  First, without memmap, just ensuring writing
        # to file works as well as to BytesIO done above.
        filename = str(tmpdir.join('a.dada'))
        with dada.open(filename, 'wb') as fw:
            fw.write_frame(frame)

        with dada.open(filename, 'rb') as fw:
            frame3 = fw.read_frame()

        assert frame3 == frame
        # Now memmap file to be written to.
        with dada.open(filename, 'wb') as fw:
            frame4 = fw.memmap_frame(frame.header)
        # Initially no data set, so frames should not match yet.
        assert frame4 != frame
        # So, if we read this file, it also should not match
        with dada.open(filename, 'rb') as fw:
            frame5 = fw.read_frame()
        assert frame5 != frame

        # Fill in some data.  This should only update some words.
        frame4[:20] = frame[:20]
        assert np.all(frame4[:20] == frame[:20])
        assert frame4 != frame
        # Update the rest, so it becomes the same.
        frame4[20:] = frame[20:]
        assert frame4 == frame
        # delete to flush to disk just to be sure, then read and check it.
        del frame4
        with dada.open(filename, 'rb') as fw:
            frame6 = fw.read_frame()

        assert frame6 == frame
        # some further tests for completeness
        # initiate frame using data and header keywords
        with dada.open(filename, 'wb') as fw:
            fw.write_frame(self.payload.data, **self.header)
        with dada.open(filename, 'rb') as fh:
            frame7 = fh.read_frame()
        assert frame7 == frame
        # memmap frame using header keywords
        with dada.open(filename, 'wb') as fw:
            frame8 = fw.memmap_frame(**self.header)
            frame8[:] = self.payload.data
        assert frame8 == frame
        del frame8
        with dada.open(filename, 'rb') as fh:
            frame9 = fh.read_frame()
        assert frame9 == frame

    def test_filestreamer(self, tmpdir):
        start_time = self.header.time
        with dada.open(SAMPLE_FILE, 'rs') as fh:
            assert fh.header0 == self.header
            assert fh.size == 16000
            assert fh.start_time == start_time
            assert fh.sample_rate == 16 * u.MHz
            record1 = fh.read(12)
            assert fh.tell() == 12
            fh.seek(10000)
            record2 = np.zeros((2, 2), dtype=np.complex64)
            record2 = fh.read(out=record2)
            assert fh.tell() == 10002
            assert fh.time == fh.tell(unit='time')
            assert (np.abs(fh.time - (start_time + 10002 / (16 * u.MHz))) <
                    1. * u.ns)
            fh.seek(fh.start_time + 1000 / (16*u.MHz))
            assert fh.tell() == 1000
            assert fh._last_header == fh.header0
            assert np.abs(fh.stop_time -
                          (start_time + 16000 / (16.*u.MHz))) < 1.*u.ns
            # Test seeker works with both int and str values for whence
            assert fh.seek(13, 0) == fh.seek(13, 'start')
            assert fh.seek(-13, 2) == fh.seek(-13, 'end')
            fhseek_int = fh.seek(17, 1)
            fh.seek(-17, 'current')
            fhseek_str = fh.seek(17, 'current')
            assert fhseek_int == fhseek_str
            with pytest.raises(ValueError):
                fh.seek(0, 'last')

        assert record1.shape == (12, 2)
        assert np.all(record1[:3] == np.array(
            [[-38.-38.j, -38.-38.j],
             [-38.-38.j, -40.+0.j],
             [-105.+60.j, 85.-15.j]], dtype=np.complex64))
        assert record1.shape == (12, 2) and record1.dtype == np.complex64
        assert np.all(record1 == self.payload[:12].squeeze())
        assert record2.shape == (2, 2)
        assert np.all(record2 == self.payload[10000:10002].squeeze())

        filename = str(tmpdir.join('a.dada'))
        with dada.open(filename, 'ws', header=self.header,
                       squeeze=False) as fw:
            assert fw.sample_rate == 16 * u.MHz
            fw.write(self.payload.data)
            assert fw.start_time == start_time
            assert (np.abs(fw.time - (start_time + 16000 / (16. * u.MHz))) <
                    1. * u.ns)

        with dada.open(filename, 'rs') as fh:
            data = fh.read()
            assert fh.start_time == start_time
            assert (np.abs(fh.time - (start_time + 16000 / (16. * u.MHz))) <
                    1. * u.ns)
            assert fh.stop_time == fh.time
            assert fh.sample_rate == 16 * u.MHz
        assert np.all(data == self.payload.data.squeeze())
        # Try single polarisation, and check initialisation by header keywords.
        h = self.header
        with dada.open(filename, 'ws', time=h.time, bps=h.bps,
                       complex_data=h.complex_data, sample_rate=h.sample_rate,
                       payloadsize=32000, nthread=1, nchan=1) as fw:
            fw.write(self.payload.data[:, 0, 0])
            assert np.abs(fw.start_time - start_time) < 1.*u.ns
            assert (np.abs(fw.time - (start_time + 16000 / (16. * u.MHz))) <
                    1. * u.ns)

        with dada.open(filename, 'rs') as fh:
            data = fh.read()
            assert np.abs(fh.start_time - start_time) < 1.*u.ns
            assert np.abs(fh.stop_time - (start_time + 16000 / (16.*u.MHz))) < 1.*u.ns
        assert np.all(data == self.payload.data[:, 0, 0])

        # Try reading a single polarization.
        with dada.open(SAMPLE_FILE, 'rs', thread_ids=[0]) as fh:
            record3 = fh.read(12)
            assert np.all(record3 == record1[:12, 0])

        # Test that squeeze attribute works on read (including in-place read)
        # and write, but can be turned off if needed.
        with dada.open(SAMPLE_FILE, 'rs') as fh:
            assert fh.sample_shape == (2,)
            assert fh.sample_shape.npol == 2
            assert fh.read(1).shape == (1, 2)
            assert fh.read(10).shape == (10, 2)
            fh.seek(0)
            out = np.zeros((12, 2), dtype=np.complex64)
            fh.read(out=out)
            assert fh.tell() == 12
            assert np.all(out == record1)
            fh.squeeze = False
            assert fh.sample_shape == (2, 1)
            assert fh.sample_shape.npol == 2
            assert fh.sample_shape.nchan == 1
            assert fh.read(1).shape == (1, 2, 1)
            assert fh.read(10).shape == (10, 2, 1)
            fh.seek(0)
            out = np.zeros((12, 2, 1), dtype=np.complex64)
            fh.read(out=out)
            assert fh.tell() == 12
            assert np.all(out.squeeze() == record1)

        with dada.open(filename, 'ws', header=self.header) as fw:
            assert fw.sample_shape == (2,)
            assert fw.sample_shape.npol == 2
            fw.write(self.payload[:8000].squeeze())
            fw.squeeze = False
            assert fw.sample_shape == (2, 1)
            assert fw.sample_shape.npol == 2
            assert fw.sample_shape.nchan == 1
            fw.write(self.payload[8000:16000])

        with dada.open(filename, 'rs', squeeze=False) as fh:
            assert np.all(fh.read() == self.payload)

    # Test that writing an incomplete stream is possible, and that frame set is
    # valid but invalid samples are appropriately marked.
    def test_incomplete_stream(self, tmpdir):
        filename = str(tmpdir.join('a.dada'))
        with catch_warnings(UserWarning) as w:
            with dada.open(filename, 'ws', header=self.header,
                           squeeze=False) as fw:
                fw.write(self.payload[:10])
        assert len(w) == 1
        assert 'partial buffer' in str(w[0].message)
        with dada.open(filename, 'rs', squeeze=False) as fwr:
            assert fwr._frame.valid
            data = fwr.read()
            assert np.all(data[:10] == self.payload[:10])
            assert np.all(data[10:] == fwr._frame.invalid_data_value)

    def test_multiple_files_stream(self, tmpdir):
        start_time = self.header.time
        data = self.payload.data.squeeze()
        header = self.header.copy()
        header.payloadsize = self.header.payloadsize // 2
        filenames = (str(tmpdir.join('a.dada')),
                     str(tmpdir.join('b.dada')))
        with dada.open(filenames, 'ws', header=header) as fw:
            start_time = fw.start_time
            fw.write(data[:1000])
            time1000 = fw.time
            fw.write(data[1000:])
            stop_time = fw.time
        assert start_time == header.time
        assert np.abs(time1000 - (start_time + 1000 / (16.*u.MHz))) < 1.*u.ns
        assert np.abs(stop_time - (start_time + 16000 / (16.*u.MHz))) < 1.*u.ns

        with dada.open(filenames[1], 'rs') as fr:
            assert (np.abs(fr.time - (start_time + 8000 / (16. * u.MHz))) <
                    1. * u.ns)
            data1 = fr.read()
        assert np.all(data1 == data[8000:])

        with dada.open(filenames, 'rs') as fr:
            assert fr.start_time == start_time
            assert fr.time == start_time
            assert np.abs(fr.stop_time -
                          (start_time + 16000 / (16. * u.MHz))) < 1. * u.ns
            data2 = fr.read()
            assert fr.time == fr.stop_time
        assert np.all(data2 == data)

    def test_template_stream(self, tmpdir):
        start_time = self.header.time
        data = self.payload.data.squeeze()
        header = self.header.copy()
        header.payloadsize = self.header.payloadsize // 4
        template = str(tmpdir.join('a{frame_nr}.dada'))
        with dada.open(template, 'ws', header=header) as fw:
            fw.write(data[:1000])
            time1000 = fw.time
            fw.write(data[1000:])
            stop_time = fw.time
        assert np.abs(time1000 - (header.time + 1000 / (16.*u.MHz))) < 1.*u.ns
        assert np.abs(stop_time - (header.time + 16000 /
                                   (16. * u.MHz))) < 1. * u.ns

        with dada.open(template.format(frame_nr=1), 'rs') as fr:
            data1 = fr.read()
            assert fr.time == fr.stop_time
            assert np.abs(fr.start_time -
                          (start_time + 4000 / (16.*u.MHz))) < 1.*u.ns
            assert np.abs(fr.stop_time -
                          (start_time + 8000 / (16.*u.MHz))) < 1.*u.ns
        assert np.all(data1 == data[4000:8000])

        with dada.open(template, 'rs') as fr:
            assert fr.time == start_time
            data2 = fr.read()
            assert fr.stop_time == fr.time
            assert np.abs(fr.stop_time -
                          (header.time + 16000 / (16. * u.MHz))) < 1. * u.ns
        assert np.all(data2 == data)

        # More complicated template, 8 files.
        header.payloadsize = self.header.payloadsize // 8
        template = str(tmpdir
                       .join('{utc_start}_{obs_offset:016d}.000000.dada'))
        with dada.open(template, 'ws', header=header) as fw:
            fw.write(data[:7000])
            assert fw.start_time == header.time
            assert (np.abs(fw.time - (start_time + 7000 / (16. * u.MHz))) <
                    1. * u.ns)
            assert fw._frame_nr == 3
            fw.write(data[7000:])
            assert (np.abs(fw.time - (start_time + 16000 / (16. * u.MHz))) <
                    1. * u.ns)

        name3 = template.format(utc_start=header['UTC_START'],
                                obs_offset=header['OBS_OFFSET'] +
                                3 * header.payloadsize)
        with dada.open(name3, 'rs') as fr:
            assert np.abs(fr.start_time -
                          (start_time + 6000 / (16.*u.MHz))) < 1.*u.ns
            assert np.abs(fr.stop_time -
                          (start_time + 8000 / (16.*u.MHz))) < 1.*u.ns
            data1 = fr.read()
            assert fr.stop_time == fr.time
        assert np.all(data1 == data[6000:8000])

        # we cannot just open using the same template, since UTC_START is
        # not available.
        with pytest.raises(KeyError):
            dada.open(template, 'rs')

        kwargs = dict(UTC_START=header['UTC_START'],
                      OBS_OFFSET=header['OBS_OFFSET'] +
                      3 * header.payloadsize,
                      FILE_SIZE=header['FILE_SIZE'])
        with dada.open(template, 'rs', **kwargs) as fr:
            assert (np.abs(fr.time - (start_time + 6000 / (16. * u.MHz))) <
                    1. * u.ns)
            data2 = fr.read()
            assert fr.time == fr.stop_time
            assert np.abs(fr.stop_time -
                          (start_time + 16000 / (16.*u.MHz))) < 1.*u.ns
        assert np.all(data2 == data[6000:])

        # just to check internal checks are OK.
        with pytest.raises(ValueError):
            dada.open(name3, 's')
        with pytest.raises(TypeError):
            # extraneous argument
            dada.open(name3, 'rs', files=(name3,))


class TestDADAFileNameSequencer(object):
    def setup(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            self.header = dada.DADAHeader.fromfile(fh)

    def test_basic_enumeration(self):
        fns1 = DADAFileNameSequencer('x{file_nr:03d}.dada', {})
        assert fns1[0] == 'x000.dada'
        assert fns1[100] == 'x100.dada'
        fns2 = DADAFileNameSequencer('{snake}_{frame_nr}', {'SNAKE': 'python'})
        assert fns2[10] == 'python_10'
        fns3 = DADAFileNameSequencer('{obs_offset:06d}.x', {'OBS_OFFSET': 10,
                                                            'FILE_SIZE': 20})
        assert fns3[0] == '000010.x'
        assert fns3[9] == '000190.x'

        with pytest.raises(KeyError):
            DADAFileNameSequencer('{snake:06d}.x', {'PYTHON': 10})

        with pytest.raises(KeyError):
            DADAFileNameSequencer('{obs_offset:06d}.x', {'OBS_OFFSET': 10})

    def test_header_enumeration(self):
        template = '{frame_nr}_{obs_offset:016d}.dada'
        fns = DADAFileNameSequencer(template, self.header)
        assert fns[0] == '0_0000006400000000.dada'
        assert fns[1] == '1_0000006400064000.dada'
        assert fns[10] == '10_0000006400640000.dada'

    def test_complicated_enumeration(self):
        # Follow the typical naming scheme:
        # 2016-04-23-07:29:30_0000000000000000.000000.dada
        template = '{utc_start}_{obs_offset:016d}.000000.dada'
        fns = DADAFileNameSequencer(template, self.header)
        assert fns[0] == '2013-07-02-01:37:40_0000006400000000.000000.dada'
        assert fns[100] == '2013-07-02-01:37:40_0000006406400000.000000.dada'

    def test_len(self, tmpdir):
        template = str(tmpdir.join('a{frame_nr}.dada'))
        fns = DADAFileNameSequencer(template, {})
        for i in range(5):
            assert len(fns) == i
            filename = fns[i]
            assert filename.endswith('a{}.dada'.format(i))
            with open(filename, 'wb') as fh:
                fh.write(b'bird')
        assert len(fns) == 5
        assert fns[-2] == fns[3]
        assert fns[-1].endswith('a4.dada')
        with pytest.raises(IndexError):
            fns[-10]
