import warnings

__all__ = ['bcd_decode', 'bcd_encode', 'get_frame_rate']


def bcd_decode(value):
    bcd = value
    result = 0
    factor = 1
    while bcd > 0:
        digit = bcd & 0xf
        if not (0 <= digit <= 9):
            raise ValueError("Invalid BCD encoded value {0}={1}."
                             .format(value, hex(value)))
        result += digit * factor
        factor *= 10
        bcd >>= 4
    return result


def bcd_encode(value):
    result = 0
    factor = 1
    while value > 0:
        value, digit = divmod(value, 10)
        result += digit*factor
        factor *= 16
    return result


def get_frame_rate(fh, header_class):
    """Returns the number of frames in one second of data."""
    fh.seek(0)
    header = header_class.fromfile(fh)
    assert header['frame_nr'] == 0
    sec0 = header.seconds
    while header['frame_nr'] == 0:
        fh.seek(header.payloadsize, 1)
        header = header_class.fromfile(fh)
    while header['frame_nr'] > 0:
        max_frame = header['frame_nr']
        fh.seek(header.payloadsize, 1)
        header = header_class.fromfile(fh)

    if header.seconds != sec0 + 1:
        warnings.warn("Header time changed by more than 1 second?")

    return max_frame + 1
