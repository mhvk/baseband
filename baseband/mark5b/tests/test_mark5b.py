# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.tests.helper import catch_warnings
from ... import mark5b
from ...vlbi_base.encoding import OPTIMAL_2BIT_HIGH
from ...data import SAMPLE_MARK5B as SAMPLE_FILE


# Check code on 2015-MAY-08.
# m5d /raw/mhvk/scintillometry/gp052d_wb_no0001 Mark5B-512-8-2 10
# ----> first 10016*4 bytes -> sample.m5b
# Mark5 stream: 0x256d140
#   stream = File-1/1=gp052a_wb_no0001
#   format = Mark5B-512-8-2 = 2
#   start mjd/sec = 821 19801.000000000
#   frame duration = 156250.00 ns
#   framenum = 0
#   sample rate = 32000000 Hz
#   offset = 0
#   framebytes = 10016 bytes
#   datasize = 10000 bytes
#   sample granularity = 1
#   frame granularity = 1
#   gframens = 156250
#   payload offset = 16
#   read position = 0
#   data window size = 1048576 bytes
# -3 -1  1 -1  3 -3 -3  3
# -3  3 -1  3 -1 -1 -1  1
#  3 -1  3  3  1 -1  3 -1
# Compare with my code:
# fh = Mark5BData(['/raw/mhvk/scintillometry/gp052d_wb_no0001'],
#                  channels=None, fedge=0, fedge_at_top=True)
# 'Start time: ', '2014-06-13 05:30:01.000' -> correct
# fh.header0
# <Mark5BFrameHeader sync_pattern: 0xabaddeed,
#                    year: 11,
#                    user: 3757,
#                    internal_tvg: False,
#                    frame_nr: 0,
#                    bcd_jday: 0x821,
#                    bcd_seconds: 0x19801,
#                    bcd_fraction: 0x0,
#                    crc: 0x975d>
# fh.record_read(6).astype(int)
# array([[-3, -1,  1, -1,  3, -3, -3,  3],
#        [-3,  3, -1,  3, -1, -1, -1,  1],
#        [ 3, -1,  3,  3,  1, -1,  3, -1]])


class TestMark5B(object):
    def test_header(self, tmpdir):
        with open(SAMPLE_FILE, 'rb') as fh:
            header = mark5b.Mark5BHeader.fromfile(fh, kday=56000)
        assert header.size == 16
        assert header.kday == 56000.
        assert header.jday == 821
        mjd, frac = divmod(header.time.mjd, 1)
        assert mjd == 56821
        assert round(frac * 86400) == 19801
        assert header.payloadsize == 10000
        assert header.framesize == 10016
        assert header['frame_nr'] == 0
        with open(str(tmpdir.join('test.m5b')), 'w+b') as s:
            header.tofile(s)
            s.seek(0)
            header2 = mark5b.Mark5BHeader.fromfile(s, header.kday)
        assert header2 == header
        header3 = mark5b.Mark5BHeader.fromkeys(header.kday, **header)
        assert header3 == header
        # Try initialising with properties instead of keywords.
        # Here, we let year, bcd_jday, bcd_seconds, and bcd_fraction be
        # set by giving the time, and let the crc be calculated from those.
        header4 = mark5b.Mark5BHeader.fromvalues(
            time=header.time,
            user=header['user'], internal_tvg=header['internal_tvg'],
            frame_nr=header['frame_nr'])
        assert header4 == header
        # Check that passing approximate ref_time is equivalent to passing
        # exact kday.
        with open(SAMPLE_FILE, 'rb') as fh:
            header5 = mark5b.Mark5BHeader.fromfile(
                fh, ref_time=Time(57200, format='mjd'))
        assert header5 == header
        # check payload and framesize setters
        header6 = mark5b.Mark5BHeader(header.words, kday=56000)
        header6.time == header.time
        header6.payload = 10000
        header6.framesize = 10016
        with pytest.raises(ValueError):
            header6.payloadsize = 9999
        with pytest.raises(ValueError):
            header6.framesize = 20
        # Regression tests
        header7 = header.copy()
        assert header7 == header  # This checks header.words
        # Check kday gets copied as well
        assert header7.kday == header.kday
        # Check ns rounding works correctly.
        header7.time = Time('2016-09-10T12:26:40.000000000')
        assert header7.fraction == 0.
        # Check that passing exact MJD to kday gives an error.
        with pytest.raises(AssertionError):
            mark5b.Mark5BHeader.fromkeys(56821, **header)
        # Check passing kday=None still reads the header, and we can set kday
        # afterward.
        with open(SAMPLE_FILE, 'rb') as fh:
            header8 = mark5b.Mark5BHeader.fromfile(fh, kday=None)
            assert header8.kday is None
            header8.kday = 56000
            assert header8 == header

    @pytest.mark.parametrize(('jday', 'ref_time', 'kday'),
                             [(882, Time(57500, format='mjd'), 57000),
                              (120, Time(57500, format='mjd'), 57000),
                              (882, Time(57113, format='mjd'), 56000),
                              (120, Time(57762, format='mjd'), 58000),
                              (263, Time(57762, format='mjd'), 57000),
                              (261, Time(57762, format='mjd'), 58000)])
    def test_infer_kday(self, jday, ref_time, kday):
        # Check that infer_kday returns proper kday for
        # ref_time - 500 < MJD < ref_time + 500, and uses bankers' rounding
        # at the boundaries.
        header = mark5b.header.Mark5BHeader(None, verify=False)
        header.jday = jday
        header.infer_kday(ref_time)
        assert header.kday == kday

    def test_decoding(self):
        """Check that look-up levels are consistent with mark5access."""
        o2h = OPTIMAL_2BIT_HIGH
        assert np.all(mark5b.payload.lut1bit[0] == -1.)
        assert np.all(mark5b.payload.lut1bit[0xff] == 1.)
        assert np.all(mark5b.payload.lut1bit.astype(int) ==
                      ((np.arange(256)[:, np.newaxis] >>
                        np.arange(8)) & 1) * 2 - 1)
        assert np.all(mark5b.payload.lut2bit[0] == -o2h)
        assert np.all(mark5b.payload.lut2bit[0x55] == 1.)
        assert np.all(mark5b.payload.lut2bit[0xaa] == -1.)
        assert np.all(mark5b.payload.lut2bit[0xff] == o2h)

    def test_payload(self, tmpdir):
        with open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(16)  # Skip header.
            payload = mark5b.Mark5BPayload.fromfile(fh, nchan=8, bps=2)
        assert payload._size == 10000
        assert payload.size == 10000
        assert payload.shape == (5000, 8)
        # Check sample shape validity.
        assert payload.sample_shape == (8,)
        assert payload.sample_shape.nchan == 8
        assert payload.dtype == np.float32
        assert np.all(payload[:3].astype(int) ==
                      np.array([[-3, -1, +1, -1, +3, -3, -3, +3],
                                [-3, +3, -1, +3, -1, -1, -1, +1],
                                [+3, -1, +3, +3, +1, -1, +3, -1]]))
        with open(str(tmpdir.join('test.m5b')), 'w+b') as s:
            payload.tofile(s)
            s.seek(0)
            payload2 = mark5b.Mark5BPayload.fromfile(
                s, payload.sample_shape.nchan, payload.bps)
            assert payload2 == payload
            with pytest.raises(EOFError):
                # Too few bytes.
                s.seek(100)
                mark5b.Mark5BPayload.fromfile(s, payload.sample_shape.nchan,
                                              payload.bps)

        payload3 = mark5b.Mark5BPayload.fromdata(payload.data, bps=payload.bps)
        assert payload3 == payload
        # Complex data should fail.
        with pytest.raises(ValueError):
            mark5b.Mark5BPayload(payload3.words, complex_data=True)
        with pytest.raises(ValueError):
            mark5b.Mark5BPayload.fromdata(np.zeros((5000, 8), np.complex64),
                                          bps=2)

    @pytest.mark.parametrize('item', (2, (), -1, slice(1, 3), slice(2, 4),
                                      slice(2, 4), slice(-3, None),
                                      (2, slice(3, 5)), (10, 4),
                                      (slice(None), 5)))
    def test_payload_getitem_setitem(self, item):
        with open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(16)  # Skip header.
            payload = mark5b.Mark5BPayload.fromfile(fh, nchan=8, bps=2)
        sel_data = payload.data[item]
        assert np.all(payload[item] == sel_data)
        payload2 = mark5b.Mark5BPayload(payload.words.copy(), nchan=8, bps=2)
        assert payload2 == payload
        payload2[item] = -sel_data
        check = payload.data
        check[item] = -sel_data
        assert np.all(payload2[item] == -sel_data)
        assert np.all(payload2.data == check)
        assert payload2 != payload
        payload2[item] = sel_data
        assert np.all(payload2[item] == sel_data)
        assert payload2 == payload

    def test_frame(self, tmpdir):
        with mark5b.open(SAMPLE_FILE, 'rb', kday=56000, nchan=8, bps=2) as fh:
            header = mark5b.Mark5BHeader.fromfile(fh, kday=56000)
            payload = mark5b.Mark5BPayload.fromfile(fh, nchan=8, bps=2)
            fh.seek(0)
            frame = fh.read_frame()

        assert frame.header == header
        assert frame.payload == payload
        assert frame == mark5b.Mark5BFrame(header, payload)
        assert np.all(frame.data[:3].astype(int) ==
                      np.array([[-3, -1, +1, -1, +3, -3, -3, +3],
                                [-3, +3, -1, +3, -1, -1, -1, +1],
                                [+3, -1, +3, +3, +1, -1, +3, -1]]))
        with open(str(tmpdir.join('test.m5b')), 'w+b') as s:
            frame.tofile(s)
            s.seek(0)
            frame2 = mark5b.Mark5BFrame.fromfile(s, kday=56000,
                                                 nchan=frame.shape[1],
                                                 bps=frame.payload.bps)
        assert frame2 == frame

        # Check passing in reference time.
        with mark5b.open(SAMPLE_FILE, 'rb', nchan=8, bps=2,
                         ref_time=Time('2014-06-13 12:00:00')) as fh:
            frame3 = fh.read_frame()
        assert frame3 == frame
        with mark5b.open(SAMPLE_FILE, 'rb', nchan=8, bps=2,
                         ref_time=Time('2015-12-13 12:00:00')) as fh:
            frame4 = fh.read_frame()
        assert frame4 == frame

        frame5 = mark5b.Mark5BFrame.fromdata(payload.data, header, bps=2)
        assert frame5 == frame
        frame6 = mark5b.Mark5BFrame.fromdata(payload.data, kday=56000,
                                             bps=2, **header)
        assert frame6 == frame
        assert frame6.time == frame.time
        frame7 = mark5b.Mark5BFrame(header, payload, valid=False)
        assert frame7.valid is False
        assert np.all(frame7.data == 0.)
        frame7.valid = True
        assert frame7 == frame
        frame8 = mark5b.Mark5BFrame.fromdata(payload.data, header, bps=2,
                                             valid=False)
        assert frame8.valid is False
        assert np.all(frame8.data == 0.)
        with open(str(tmpdir.join('test8.m5b')), 'w+b') as s:
            frame8.tofile(s)
            s.seek(0)
            frame9 = mark5b.Mark5BFrame.fromfile(s, kday=56000,
                                                 nchan=frame8.shape[1],
                                                 bps=frame8.payload.bps)
        assert frame9.valid is False
        assert np.all(frame9.data == 0.)
        assert np.all(frame9.payload.words == 0x11223344)

    def test_header_times(self):
        with mark5b.open(SAMPLE_FILE, 'rb', kday=56000, nchan=8, bps=2) as fh:
            header0 = mark5b.Mark5BHeader.fromfile(fh, kday=56000)
            start_time = header0.time
            samples_per_frame = header0.payloadsize * 8 // 2 // 8
            frame_rate = 32. * u.MHz / samples_per_frame
            frame_duration = 1. / frame_rate
            fh.seek(0)
            while True:
                try:
                    frame = fh.read_frame()
                except EOFError:
                    break
                header_time = frame.header.time
                expected = (start_time +
                            frame.header['frame_nr'] * frame_duration)
                assert abs(header_time - expected) < 1. * u.ns

        # On the last frame, also check one can recover the time if 'frac_sec'
        # is not set.
        header = frame.header.copy()
        header['bcd_fraction'] = 0
        # So, now recover first header time, which started on the second.
        assert header.time == header0.time
        # But if we pass in the correct framerate, it uses the frame_nr.
        assert abs(header.get_time(frame_rate) - frame.header.time) < 1. * u.ns

        # Check setting time using framerate.
        sample_rate = 128. * u.MHz
        samples_per_frame = 5000
        # Max frame_nr is 2**15; this sets it to 25600.
        frame_rate = sample_rate / samples_per_frame
        header.set_time(time=(start_time + 1. / frame_rate),
                        framerate=frame_rate)
        header.get_time(frame_rate)
        assert abs(header.get_time(frame_rate) -
                   start_time - 1. / frame_rate) < 1. * u.ns
        header.set_time(time=(start_time + 3921. / frame_rate),
                        framerate=frame_rate)
        assert abs(header.get_time(frame_rate) -
                   start_time - 3921. / frame_rate) < 1. * u.ns
        # Test using bcd_fraction gives us within 0.1 ms accuracy.
        assert abs(header.time - start_time - 3921. / frame_rate) < 0.1 * u.ms
        header.set_time(time=(start_time + 25599. / frame_rate),
                        framerate=frame_rate)
        assert abs(header.get_time(frame_rate) -
                   start_time - 25599. / frame_rate) < 1. * u.ns
        # Check rounding when using passing fractional frametimes.
        header.set_time(time=(start_time + 25598.53 / frame_rate),
                        framerate=frame_rate)
        assert abs(header.get_time(frame_rate) -
                   start_time - 25599. / frame_rate) < 1. * u.ns
        # Check rounding to the nearest second when less than 2 ns away.
        header.set_time(time=(start_time + 0.9 * u.ns), framerate=frame_rate)
        assert header.seconds == header0.seconds
        header.set_time(time=(start_time - 0.9 * u.ns), framerate=frame_rate)
        assert header.seconds == header0.seconds

    def test_find_header(self, tmpdir):
        # Below, the tests set the file pointer to very close to a header,
        # since otherwise they run *very* slow.  This is somehow related to
        # pytest, since speed is not a big issue running stuff on its own.
        with mark5b.open(SAMPLE_FILE, 'rb', kday=56000) as fh:
            header0 = mark5b.Mark5BHeader.fromfile(fh, kday=56000)
            fh.seek(0)
            header_0 = fh.find_header()
            assert fh.tell() == 0
            fh.seek(10000)
            header_10000f = fh.find_header(forward=True)
            assert fh.tell() == header0.framesize
            fh.seek(16)
            header_16b = fh.find_header(forward=False)
            assert fh.tell() == 0
            fh.seek(-10000, 2)
            header_m10000b = fh.find_header(forward=False)
            assert fh.tell() == 3 * header0.framesize
            fh.seek(-30, 2)
            header_end = fh.find_header(forward=True)
            assert header_end is None
        assert header_16b == header_0
        assert header_10000f['frame_nr'] == 1
        assert header_m10000b['frame_nr'] == 3
        m5_test = str(tmpdir.join('test.m5b'))
        with open(m5_test, 'wb') as s, open(SAMPLE_FILE, 'rb') as f:
            s.write(f.read(10040))
            f.seek(20000)
            s.write(f.read())
        with mark5b.open(m5_test, 'rb', kday=header0.kday) as fh:
            fh.seek(0)
            header_0 = fh.find_header()
            assert fh.tell() == 0
            fh.seek(10000)
            header_10000f = fh.find_header(forward=True)
            assert fh.tell() == header0.framesize * 2 - 9960
        # For completeness, also check a really short file...
        with open(m5_test, 'wb') as s, open(SAMPLE_FILE, 'rb') as f:
            s.write(f.read(10018))
        with mark5b.open(m5_test, 'rb') as fh:
            fh.seek(10)
            header_10 = fh.find_header(forward=False)
            assert fh.tell() == 0
        assert header_10 == header0

    def test_filestreamer(self, tmpdir):
        with open(SAMPLE_FILE, 'rb') as fh:
            header = mark5b.Mark5BHeader.fromfile(fh, kday=56000)

        with mark5b.open(SAMPLE_FILE, 'rs', sample_rate=32*u.MHz, kday=56000,
                         nchan=8, bps=2) as fh:
            assert header == fh.header0
            assert fh.samples_per_frame == 5000
            assert fh.sample_rate == 32 * u.MHz
            last_header = fh._last_header
            assert fh.size == 20000
            record = fh.read(12)
            assert fh.tell() == 12
            fh.seek(10000)
            record2 = fh.read(2)
            assert fh.tell() == 10002
            assert fh.fh_raw.tell() == 3. * header.framesize
            assert fh.time == fh.tell(unit='time')
            assert (np.abs(fh.time - (fh.start_time + 10002 / (32 * u.MHz))) <
                    1. * u.ns)
            fh.seek(fh.start_time + 1000 / (32 * u.MHz))
            assert fh.tell() == 1000
            fh.seek(-10, 2)
            assert fh.tell() == fh.size - 10
            record3 = fh.read()
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

        assert last_header['frame_nr'] == 3
        assert last_header['user'] == header['user']
        assert last_header['bcd_jday'] == header['bcd_jday']
        assert last_header['bcd_seconds'] == header['bcd_seconds']
        assert last_header['bcd_fraction'] == 4
        frate = (1. / ((last_header.time - header.time) / 3.)).to(u.Hz).value
        assert round(frate) == 6400
        assert record.shape == (12, 8)
        assert np.all(record.astype(int)[:3] ==
                      np.array([[-3, -1, +1, -1, +3, -3, -3, +3],
                                [-3, +3, -1, +3, -1, -1, -1, +1],
                                [+3, -1, +3, +3, +1, -1, +3, -1]]))
        assert record2.shape == (2, 8)
        assert np.all(record2.astype(int) ==
                      np.array([[-1, -1, -1, +3, +3, -3, +3, -1],
                                [-1, +1, -3, +3, -3, +1, +3, +1]]))
        assert record3.shape == (10, 8)

        # Check passing a time object.
        with mark5b.open(SAMPLE_FILE, 'rs', sample_rate=32*u.MHz,
                         ref_time=Time('2015-01-01'), nchan=8, bps=2) as fh:
            assert fh.header0 == header
            assert fh._last_header == last_header
        with mark5b.open(SAMPLE_FILE, 'rs', sample_rate=32*u.MHz,
                         ref_time=Time('2013-01-01'), nchan=8, bps=2) as fh:
            assert fh.header0 == header
            assert fh._last_header == last_header

        # Read only some selected threads.
        with mark5b.open(SAMPLE_FILE, 'rs', sample_rate=32*u.MHz,
                         ref_time=Time(57000, format='mjd'), nchan=8, bps=2,
                         subset=[4, 5]) as fh:
            assert fh.sample_shape == (2,)
            assert fh.subset == ([4, 5],)
            record4 = fh.read(12)
        assert np.all(record4 == record[:, 4:6])
        # Read all data and check that it can be written out.
        with mark5b.open(SAMPLE_FILE, 'rs', sample_rate=32*u.MHz, kday=56000,
                         nchan=8, bps=2) as fh:
            start_time = fh.time
            record = fh.read(20000)
            stop_time = fh.time

        m5_test = str(tmpdir.join('test.m5b'))
        with mark5b.open(m5_test, 'ws', sample_rate=32*u.MHz, nchan=8, bps=2,
                         time=start_time) as fw:
            assert fw.sample_rate == 32 * u.MHz
            # Write in pieces to ensure squeezed data can be handled,
            # And add in an invalid frame for good measure.
            fw.write(record[:11])
            fw.write(record[11:5000])
            fw.write(record[5000:10000], valid=False)
            fw.write(record[10000:])
            assert fw.time == stop_time

        with mark5b.open(m5_test, 'rs', sample_rate=32*u.MHz,
                         ref_time=Time(57000, format='mjd'),
                         nchan=8, bps=2) as fh:
            assert fh.time == start_time
            assert fh.sample_rate == 32 * u.MHz
            record2 = fh.read(20000)
            assert fh.time == stop_time
            assert np.all(record2[:5000] == record[:5000])
            assert np.all(record2[5000:10000] == 0.)
            assert np.all(record2[10000:] == record[10000:])

        # Check files can be made byte-for-byte identical.
        with mark5b.open(m5_test, 'ws', sample_rate=32*u.MHz, nchan=8,
                         bps=2, time=start_time, user=header['user'],
                         internal_tvg=header['internal_tvg'],
                         frame_nr=header['frame_nr']) as fw:
            fw.write(record)

        with open(SAMPLE_FILE, 'rb') as fr, open(m5_test, 'rb') as fs:
            orig_bytes = fr.read()
            conv_bytes = fs.read()
            assert conv_bytes == orig_bytes

        # Check if data can be read across days.  Write out sample Mark 5B
        # with fake timecode and step, and see if it can be re-read.
        time_premidnight = Time('2014:164:23:59:59')
        with mark5b.open(m5_test, 'ws', sample_rate=10*u.kHz,
                         nchan=8, bps=2, time=time_premidnight) as fw:
            fw.write(record)

        with mark5b.open(m5_test, 'rs', sample_rate=10*u.kHz, kday=56000,
                         nchan=8, bps=2) as fh:
            record5 = fh.read()     # Read across days.
            assert np.all(record5 == record)
            assert (abs(fh.time - Time('2014:165:00:00:01', precision=9)) <
                    1. * u.ns)

        # As above, but checking if data can be read across kday increments
        # (2017-09-03 is MJD 57999 and 2017-09-04 is MJD 58000).
        time_preturnover = Time('2017-09-03T23:59:59', precision=9)
        with mark5b.open(m5_test, 'ws', sample_rate=10*u.kHz,
                         nchan=8, bps=2, time=time_preturnover) as fw:
            fw.write(record)

        with mark5b.open(m5_test, 'rs', sample_rate=10*u.kHz, kday=57000,
                         nchan=8, bps=2) as fh:
            assert abs(fh.start_time - time_preturnover) < 1. * u.ns
            record5 = fh.read()     # Read across kday.
            assert np.all(record5 == record)
            assert (abs(fh.time - Time('2017-09-04T00:00:01', precision=9)) <
                    1. * u.ns)

        # Test that squeeze attribute works on read (including in-place read).
        with mark5b.open(SAMPLE_FILE, 'rs', sample_rate=32*u.MHz, kday=56000,
                         nchan=8, bps=2, subset=0) as fh:
            assert fh.sample_shape == ()
            assert fh.read(1).shape == (1,)
            fh.seek(0)
            out = np.zeros(12)
            fh.read(out=out)
            assert fh.tell() == 12
            assert np.all(out == record[:12, 0])

        with mark5b.open(SAMPLE_FILE, 'rs', sample_rate=32*u.MHz, kday=56000,
                         nchan=8, bps=2, subset=[0], squeeze=False) as fh:
            assert fh.subset == ([0],)
            assert fh.sample_shape == (1,)
            assert fh.sample_shape.nchan == 1
            assert fh.read(1).shape == (1, 1)
            fh.seek(0)
            out = np.zeros((12, 1))
            fh.read(out=out)
            assert fh.tell() == 12
            assert np.all(out.squeeze() == record[:12, 0])

        # Test that squeeze attribute works on write.
        m5_test_squeeze = str(tmpdir.join('test_squeeze.m5b'))
        with mark5b.open(m5_test_squeeze, 'ws', sample_rate=32*u.MHz, nchan=1,
                         bps=2, time=start_time) as fws:
            assert fws.sample_shape == ()
            fws.write(record[:, 0])
            # Write some dummy data to fill up the rest of the frame.
            fws.write(np.zeros(20000, dtype='float32'))
        m5_test_nosqueeze = str(tmpdir.join('test_nosqueeze.m5b'))
        with mark5b.open(m5_test_nosqueeze, 'ws', sample_rate=32*u.MHz,
                         nchan=1, bps=2, squeeze=False,
                         time=start_time) as fwns:
            assert fwns.sample_shape == (1,)
            assert fwns.sample_shape.nchan == 1
            fwns.write(record[:, 0:1])    # 0:1 to keep record 2-dimensional.
            # Write some dummy data to fill up the rest of the frame.
            fwns.write(np.zeros((20000, 1), dtype='float32'))

        with mark5b.open(m5_test_squeeze, 'rs', sample_rate=32*u.MHz,
                         kday=56000, nchan=1, bps=2) as fhs, \
                mark5b.open(m5_test_nosqueeze, 'rs', sample_rate=32*u.MHz,
                            kday=56000, nchan=1, bps=2) as fhns:
            assert np.all(fhs.read(20000) == record[:, 0])
            assert np.all(fhns.read(20000) == record[:, 0])

        # Test that sample_rate can be inferred from max frame number.
        m5_test_samplerate = str(tmpdir.join('test_samplerate.m5b'))
        sample_rate = 1. * u.MHz
        samples_per_frame = 5000
        test_time = start_time + 198. * samples_per_frame / sample_rate
        with mark5b.open(m5_test_samplerate, 'ws', sample_rate=sample_rate,
                         nchan=8, bps=2, time=test_time) as fw:
            assert fw.header0['frame_nr'] == 198
            # Check that the fourth frame has wrapped the frame counter.
            frame = fw._make_frame(3)
            frame.header['frame_nr'] == 1
            # Write 4 dummy frames for the sample rate inference check.
            fw.write(np.zeros((20000, 8), dtype='float32'))

        with mark5b.open(m5_test_samplerate, 'rs', kday=56000, nchan=8) as fh:
            assert fh.sample_rate == sample_rate
            # Might as well test some other properties
            assert abs(fh.start_time - test_time) < 1.*u.ns
            assert fh.header0['frame_nr'] == 198

    # Test that writing an incomplete stream is possible, and that frame set is
    # appropriately marked as invalid.
    @pytest.mark.parametrize('fill_value', (0., -999.))
    def test_incomplete_stream(self, tmpdir, fill_value):
        m5_incomplete = str(tmpdir.join('incomplete.m5'))
        with catch_warnings(UserWarning) as w:
            with mark5b.open(SAMPLE_FILE, 'rs', sample_rate=32*u.MHz,
                             kday=56000, nchan=8, bps=2) as fr:
                record = fr.read(10)
                with mark5b.open(m5_incomplete, 'ws', header0=fr.header0,
                                 sample_rate=32*u.MHz, nchan=8) as fw:
                    fw.write(record)
        assert len(w) == 1
        assert 'partial buffer' in str(w[0].message)
        with mark5b.open(m5_incomplete, 'rs', sample_rate=32*u.MHz, kday=56000,
                         nchan=8, bps=2, fill_value=fill_value) as fwr:
            assert fwr.fill_value == fill_value
            check = fwr.read()
            assert np.all(check == fill_value)

    def test_stream_invalid(self):
        with pytest.raises(ValueError):
            mark5b.open('ts.dat', 's')

    def test_stream_missing_kday(self):
        with pytest.raises(ValueError):
            mark5b.open(SAMPLE_FILE, 'rs', sample_rate=32*u.MHz,
                        nchan=8, bps=2)
