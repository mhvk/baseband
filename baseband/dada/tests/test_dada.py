# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
import copy
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.tests.helper import catch_warnings
from ... import dada
from ...helpers import sequentialfile as sf
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
            assert header.nbytes == 4096
            assert fh.tell() == 4096
        assert header['NDIM'] == 2
        assert header['NCHAN'] == 1
        assert header['UTC_START'] == '2013-07-02-01:37:40'
        assert header['OBS_OFFSET'] == 6400000000  # 100 s
        assert header.time.isot == '2013-07-02T01:39:20.000'
        assert header.frame_nbytes == 64000 + 4096
        assert header.payload_nbytes == 64000
        assert header.mutable is False
        with pytest.raises(TypeError):
            header['NCHAN'] = 2
        assert header['NCHAN'] == 1
        with pytest.raises(AttributeError):
            header.python

        with open(str(tmpdir.join('test.dada')), 'w+b') as s:
            header.tofile(s)
            assert s.tell() == header.nbytes
            s.seek(0)
            header2 = dada.DADAHeader.fromfile(s)
            assert header2 == header
            assert header2.mutable is False
            assert s.tell() == header.nbytes

        with open(str(tmpdir.join('test.dada')), 'w+b') as s:
            # Now create header with wrong HDR_SIZE in file.
            bad_header = header.copy()
            bad_header['HDR_SIZE'] = 1000
            with pytest.raises(ValueError):
                bad_header.tofile(s)
            # Now write header explicitly, skipping the check in tofile,
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
        # Check attribute setting.
        header3.start_time = header.start_time - 0.5 * u.day
        assert np.abs(header3.start_time -
                      (header.start_time - 0.5 * u.day)) < 1 * u.ns
        assert np.abs(header3.time - (header.time - 0.5 * u.day)) < 1 * u.ns
        # Check against rounding.
        just_below_int = Time(55000, -1e-15, format='mjd')
        header3.start_time = just_below_int
        assert header3['MJD_START'] == '54999.999999999999999'
        header3['NCHAN'] = 2
        assert header3['NCHAN'] == 2
        header3.frame_nbytes = 9096
        assert header3.payload_nbytes == 5000
        # Try initialising with properties instead of keywords.
        # Here, we first just try the start time and offset.
        header4 = dada.DADAHeader.fromvalues(
            start_time=header.start_time,
            offset=header.time - header.start_time,
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
        # And now try the time and offset.
        header5 = dada.DADAHeader.fromvalues(
            offset=header.time - header.start_time, time=header.time,
            bps=header.bps, complex_data=header.complex_data,
            sample_rate=header.sample_rate, sideband=header.sideband,
            samples_per_frame=header.samples_per_frame,
            sample_shape=header.sample_shape,
            source=header['SOURCE'], ra=header['RA'], dec=header['DEC'],
            telescope=header['TELESCOPE'], instrument=header['INSTRUMENT'],
            receiver=header['RECEIVER'], freq=header['FREQ'],
            pic_version=header['PIC_VERSION'])
        # Finally try the time and start time.
        assert header5 == header
        header6 = dada.DADAHeader.fromvalues(
            time=header.time, start_time=header.start_time,
            bps=header.bps, complex_data=header.complex_data,
            sample_rate=header.sample_rate, sideband=header.sideband,
            samples_per_frame=header.samples_per_frame,
            sample_shape=header.sample_shape,
            source=header['SOURCE'], ra=header['RA'], dec=header['DEC'],
            telescope=header['TELESCOPE'], instrument=header['INSTRUMENT'],
            receiver=header['RECEIVER'], freq=header['FREQ'],
            pic_version=header['PIC_VERSION'])
        assert header6 == header
        # Check repr can be used to instantiate header
        header7 = eval('dada.' + repr(header))
        assert header7 == header
        # repr includes the comments
        assert header7.comments == header.comments
        # Therefore repr should be identical too.
        assert repr(header7) == repr(header)
        # Check instantiation via tuple
        header8 = dada.DADAHeader(((key, (header[key], header.comments[key]))
                                   for key in header))
        assert header8 == header
        assert header8.comments == header.comments
        # Check copying
        header9 = header.copy()
        assert header9 == header
        assert header9.mutable is True
        assert header9.comments == header.comments
        header10 = copy.copy(header9)
        assert header10 == header
        assert header10.mutable is True
        assert header10.comments == header.comments

    def test_payload(self, tmpdir):
        payload = self.payload
        assert payload.nbytes == 64000
        assert payload.shape == (16000, 2, 1)
        # Check sample shape validity.
        assert payload.sample_shape == (2, 1)
        assert payload.sample_shape.npol == 2
        assert payload.sample_shape.nchan == 1
        assert payload.size == 32000
        assert payload.ndim == 3
        assert payload.dtype == np.complex64
        assert np.all(payload[:3] == np.array(
            [[[-38.-38.j], [-38.-38.j]],
             [[-38.-38.j], [-40.+0.j]],
             [[-105.+60.j], [85.-15.j]]], dtype=np.complex64))

        with open(str(tmpdir.join('test.dada')), 'w+b') as s:
            payload.tofile(s)
            s.seek(0)
            payload2 = dada.DADAPayload.fromfile(s, payload_nbytes=64000,
                                                 sample_shape=(2, 1), bps=8,
                                                 complex_data=True)
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

    def test_file_reader(self):
        with dada.open(SAMPLE_FILE, 'rb') as fh:
            header = fh.read_header()
            assert header == self.header
            current_pos = fh.tell()
            frame_rate = fh.get_frame_rate()
            assert frame_rate == header.sample_rate / header.samples_per_frame
            assert fh.tell() == current_pos
            # Frame is done below, as is writing in binary.

    def test_frame(self, tmpdir):
        with dada.open(SAMPLE_FILE, 'rb') as fh:
            frame = fh.read_frame(memmap=False)
        header, payload = frame.header, frame.payload
        assert header == self.header
        assert payload == self.payload
        assert frame == dada.DADAFrame(header, payload)
        assert frame.shape == payload.shape
        assert frame.size == payload.size
        assert frame.ndim == payload.ndim
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
        # Some further tests for completeness;
        # initiate frame using data and header keywords.
        with dada.open(filename, 'wb') as fw:
            fw.write_frame(self.payload.data, **self.header)
        with dada.open(filename, 'rb') as fh:
            frame7 = fh.read_frame()
        assert frame7 == frame
        # memmap frame using header keywords.
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
            assert fh.sample_shape == (2,)
            assert fh.shape == (16000,) + fh.sample_shape
            assert fh.size == np.prod(fh.shape)
            assert fh.ndim == len(fh.shape)
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
            # Test seeker works with both int and str values for whence.
            assert fh.seek(13, 0) == fh.seek(13, 'start')
            assert fh.seek(-13, 2) == fh.seek(-13, 'end')
            fhseek_int = fh.seek(17, 1)
            fh.seek(-17, 'current')
            fhseek_str = fh.seek(17, 'current')
            assert fhseek_int == fhseek_str
            with pytest.raises(ValueError):
                fh.seek(0, 'last')
            fh.seek(1, 'end')
            with pytest.raises(EOFError):
                fh.read()

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
        with dada.open(filename, 'ws', header0=self.header,
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
                       payload_nbytes=32000, npol=1, nchan=1) as fw:
            fw.write(self.payload.data[:, 0, 0])
            assert np.abs(fw.start_time - start_time) < 1.*u.ns
            assert (np.abs(fw.time - (start_time + 16000 / (16. * u.MHz))) <
                    1. * u.ns)

        with dada.open(filename, 'rs') as fh:
            data_onepol = fh.read()
            assert np.abs(fh.start_time - start_time) < 1.*u.ns
            assert np.abs(fh.stop_time - (start_time + 16000 /
                                          (16.*u.MHz))) < 1.*u.ns
        assert np.all(data_onepol == self.payload.data[:, 0, 0])

        # Try reading a single polarization.
        with dada.open(SAMPLE_FILE, 'rs', subset=0) as fh:
            assert fh.sample_shape == ()
            assert fh.subset == (0,)
            record3 = fh.read(12)
            assert np.all(record3 == record1[:12, 0])

        # Create an npol=2, nchan=2 file and subset it.
        data2d = np.array([data, -data]).transpose(1, 2, 0)
        with dada.open(filename, 'ws', time=h.time, bps=h.bps,
                       complex_data=h.complex_data, sample_rate=h.sample_rate,
                       payload_nbytes=32000, npol=2, nchan=2) as fw:
            fw.write(data2d)
            assert np.abs(fw.start_time - start_time) < 1.*u.ns
            assert (np.abs(fw.time - (start_time + 16000 / (16. * u.MHz))) <
                    1. * u.ns)

        # First check if write was successful.
        with dada.open(filename, 'rs') as fh:
            assert fh.sample_shape == (2, 2)
            data_all = fh.read()
            assert np.all(data_all == data2d)

        # Then read right polarization, but read channels in reverse order.
        with dada.open(filename, 'rs', subset=(1, [1, 0])) as fh:
            assert fh.sample_shape == (2,)
            data_sub = fh.read()
            assert np.all(data_sub[:, 1] == data2d[:, 1, 0])

        # Test that squeeze attribute works on read (including in-place read;
        # we implicitly tested squeeze=True above).
        with dada.open(SAMPLE_FILE, 'rs', squeeze=False) as fh:
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

        # Test that squeeze attribute works on write.
        dada_test_squeeze = str(tmpdir.join('test_squeeze.dada'))
        with dada.open(dada_test_squeeze, 'ws', header0=self.header) as fw:
            assert fw.sample_shape == (2,)
            assert fw.sample_shape.npol == 2
            fw.write(self.payload.data.squeeze())
        dada_test_nosqueeze = str(tmpdir.join('test_nosqueeze.dada'))
        with dada.open(dada_test_nosqueeze, 'ws', header0=self.header,
                       squeeze=False) as fw:
            assert fw.sample_shape == (2, 1)
            assert fw.sample_shape.npol == 2
            assert fw.sample_shape.nchan == 1
            fw.write(self.payload.data)

        with dada.open(dada_test_squeeze, 'rs', squeeze=False) as fhs, \
                dada.open(dada_test_nosqueeze, 'rs', squeeze=False) as fhns:
            assert np.all(fhs.read() == self.payload)
            assert np.all(fhns.read() == self.payload)

        # Test passing squeeze=False along with header keywords (otherwise
        # identical to test above).
        h = self.header

        with dada.open(filename, 'ws', time=h.time,
                       payload_nbytes=h.payload_nbytes,
                       sample_rate=h.sample_rate, complex_data=h.complex_data,
                       sample_shape=h.sample_shape, bps=h.bps,
                       squeeze=False) as fw:
            assert fw.sample_shape == (2, 1)
            assert fw.sample_shape.npol == 2
            assert fw.sample_shape.nchan == 1
            fw.write(self.payload.data)

        with dada.open(dada_test_squeeze, 'rs', squeeze=False) as fhs, \
                dada.open(dada_test_nosqueeze, 'rs', squeeze=False) as fhns:
            assert np.all(fhs.read() == self.payload)
            assert np.all(fhns.read() == self.payload)

    def test_incomplete_stream(self, tmpdir):
        """Test that writing an incomplete stream is possible, and that frame
        set is valid but invalid samples use the fill value.
        """
        filename = str(tmpdir.join('a.dada'))
        with catch_warnings(UserWarning) as w:
            with dada.open(filename, 'ws', header0=self.header,
                           squeeze=False) as fw:
                fw.write(self.payload[:10])
        assert len(w) == 1
        assert 'partial buffer' in str(w[0].message)
        with dada.open(filename, 'rs', squeeze=False) as fwr:
            data = fwr.read()
            assert np.all(data[:10] == self.payload[:10])
            assert np.all(data[10:] == fwr.fill_value)

    def test_multiple_files_stream(self, tmpdir):
        start_time = self.header.time
        data = self.payload.data.squeeze()
        header = self.header.copy()
        header.payload_nbytes = self.header.payload_nbytes // 2
        filenames = (str(tmpdir.join('a.dada')),
                     str(tmpdir.join('b.dada')))
        with dada.open(filenames, 'ws', **header) as fw:
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

        # Pass sequentialfile objects to reader.
        with sf.open(filenames, 'w+b',
                     file_size=(header.payload_nbytes + 4096)) as fraw, \
                dada.open(fraw, 'ws', header0=header) as fw:
            fw.write(data)

        with dada.open(filenames, 'rs') as fr:
            data3 = fr.read()
        assert np.all(data3 == data)

        # Test passing stream reader kwargs.
        with dada.open(filenames, 'rs', subset=1, squeeze=False) as fr:
            data4 = fr.read()
        assert np.all(data4.squeeze() == data[:, 1])

        # Check that we can't pass a filename sequence in 'wb' mode.
        with pytest.raises(ValueError):
            dada.open(filenames, 'wb')

    def test_partial_last_frame(self, tmpdir):
        """Test reading an incomplete frame from one or a sequence of files."""
        # Prepare file sequence.
        header = self.header.copy()
        data = self.payload.data.squeeze()
        data = np.r_[data, data, data]
        filenames = [str(tmpdir.join('a.dada')),
                     str(tmpdir.join('b.dada')),
                     str(tmpdir.join('c.dada'))]
        with dada.open(filenames, 'ws', header0=header) as fw:
            fw.write(data)

        # Replace c.dada with partially complete file.
        with dada.open(str(tmpdir.join('c.dada')), 'rb') as fh, \
                dada.open(str(tmpdir.join('c_partial.dada')), 'wb') as fw:
            full_filesize = fh.seek(0, 2)
            fh.seek(0)
            fw.write(fh.read(full_filesize // 2 - 37))

        filenames[-1] = str(tmpdir.join('c_partial.dada'))

        # Check reading single partial frame.
        with dada.open(filenames[-1], 'rs') as fh:
            # Check that we've written the right number of bytes to file.
            filesize = fh.fh_raw.seek(0, 2)
            fh.fh_raw.seek(0)
            assert filesize == full_filesize // 2 - 37
            # Payload truncates 3 bytes so there is an integer number of
            # complete samples.
            assert fh.header0.frame_nbytes == filesize - 3
            assert fh.header0.nbytes == self.header.nbytes
            assert fh.samples_per_frame == (
                (filesize - self.header.nbytes) * 8 // fh.header0.bps // 2 //
                np.prod(fh.header0.sample_shape))
            assert fh.header0 is fh._last_header
            assert np.abs(fh.stop_time - fh.start_time -
                          7478 / fh.sample_rate) < 1 * u.ns
            assert fh.shape == (7478, 2)
            # Taking advantage of data being repeated 3 times.
            assert np.all(fh.read() == data[:7478])

        # Check reading sequence of files.
        with dada.open(filenames) as fh:
            assert fh.samples_per_frame == header.samples_per_frame
            assert np.abs(fh.stop_time - fh.start_time -
                          39478 / fh.sample_rate) < 1 * u.ns
            assert fh.shape == (39478, 2)
            assert np.all(fh.read() == data[:39478])
            fh.seek(-29, 2)
            assert np.all(fh.read() == data[7478 - 29:7478])
            assert fh.tell() == 39478

        # Replace c.dada with only the header (and no payload).
        with dada.open(str(tmpdir.join('c.dada')), 'rb') as fh, \
                dada.open(str(tmpdir.join('c_header_only.dada')), 'wb') as fw:
            fw.write(fh.read(4096))
            fh.seek(0)
            header_c = fh.read_header()

        filenames[-1] = str(tmpdir.join('c_header_only.dada'))

        # Check that reading the frame gives payload of zero size.
        with dada.open(str(tmpdir.join('c_header_only.dada')), 'rb') as fp:
            assert fp.read_header() == header_c
            fp.seek(0)
            with pytest.raises(ValueError) as excinfo:
                fp.read_frame()
            assert "mmap length is greater" in str(excinfo.value)

        with pytest.raises(EOFError) as excinfo:
            with dada.open(str(tmpdir.join('c_header_only.dada')), 'rs') as fp:
                pass
        assert "appears to end without" in str(excinfo.value)

        # Reading the new sequence, the last frame should be ignored.
        with dada.open(filenames) as fh:
            assert np.abs(fh.stop_time - fh.start_time -
                          32000 / fh.sample_rate) < 1 * u.ns
            assert fh.shape == (32000, 2)

    def test_template_stream(self, tmpdir):
        start_time = self.header.time
        data = self.payload.data.squeeze()
        header = self.header.copy()
        header.payload_nbytes = self.header.payload_nbytes // 4
        template = str(tmpdir.join('a{frame_nr}.dada'))
        with dada.open(template, 'ws', header0=header) as fw:
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
        header.payload_nbytes = self.header.payload_nbytes // 8
        template = str(tmpdir
                       .join('{utc_start}_{obs_offset:016d}.000000.dada'))
        with dada.open(template, 'ws', header0=header) as fw:
            fw.write(data[:7000])
            assert fw.start_time == header.time
            assert (np.abs(fw.time - (start_time + 7000 / (16. * u.MHz))) <
                    1. * u.ns)
            fw.write(data[7000:])
            assert (np.abs(fw.time - (start_time + 16000 / (16. * u.MHz))) <
                    1. * u.ns)

        name3 = template.format(utc_start=header['UTC_START'],
                                obs_offset=header['OBS_OFFSET'] +
                                3 * header.payload_nbytes)
        with dada.open(name3, 'rs') as fr:
            assert np.abs(fr.start_time -
                          (start_time + 6000 / (16.*u.MHz))) < 1.*u.ns
            assert np.abs(fr.stop_time -
                          (start_time + 8000 / (16.*u.MHz))) < 1.*u.ns
            data1 = fr.read()
            assert fr.stop_time == fr.time
        assert np.all(data1 == data[6000:8000])

        # We cannot just open using the same template, since UTC_START is
        # not available.
        with pytest.raises(KeyError):
            dada.open(template, 'rs')

        kwargs = dict(UTC_START=header['UTC_START'],
                      OBS_OFFSET=header['OBS_OFFSET'] +
                      3 * header.payload_nbytes,
                      FILE_SIZE=header['FILE_SIZE'])
        with dada.open(template, 'rs', **kwargs) as fr:
            assert (np.abs(fr.time - (start_time + 6000 / (16. * u.MHz))) <
                    1. * u.ns)
            data2 = fr.read()
            assert fr.time == fr.stop_time
            assert np.abs(fr.stop_time -
                          (start_time + 16000 / (16.*u.MHz))) < 1.*u.ns
        assert np.all(data2 == data[6000:])

        # Just to check internal checks are OK.
        with pytest.raises(ValueError):
            dada.open(name3, 's')
        with pytest.raises(TypeError):
            # Extraneous argument.
            dada.open(name3, 'rs', files=(name3,))


class TestDADAFileNameSequencer(object):
    def setup(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            self.header = dada.DADAHeader.fromfile(fh)

    def test_offset_enumeration(self):
        fns = DADAFileNameSequencer(
            '{obs_offset:06d}.x', {'OBS_OFFSET': 10, 'FILE_SIZE': 20})
        assert fns[0] == '000010.x'
        assert fns[9] == '000190.x'

        with pytest.raises(KeyError):
            DADAFileNameSequencer('{obs_offset:06d}.x', {'OBS_OFFSET': 10})

    def test_complicated_enumeration(self):
        # Tests that frame_nr properly draws from file_nr.
        template = '{frame_nr}_{obs_offset:016d}.dada'
        fns = DADAFileNameSequencer(template, self.header)
        assert fns[0] == '0_0000006400000000.dada'
        assert fns[1] == '1_0000006400064000.dada'
        assert fns[10] == '10_0000006400640000.dada'

        # Follow the typical naming scheme:
        # 2016-04-23-07:29:30_0000000000000000.000000.dada
        template = '{utc_start}_{obs_offset:016d}.000000.dada'
        fns = DADAFileNameSequencer(template, self.header)
        assert fns[0] == '2013-07-02-01:37:40_0000006400000000.000000.dada'
        assert fns[100] == '2013-07-02-01:37:40_0000006406400000.000000.dada'
