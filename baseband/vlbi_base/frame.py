# Helper functions for VLBI readers (VDIF, Mark5B).
import io
import numpy as np


class VLBIFrameBase(object):

    _header_class = None
    _payload_class = None

    def __init__(self, header, payload, valid=True, verify=True):
        self.header = header
        self.payload = payload
        self.valid = valid
        if verify:
            self.verify()

    def verify(self):
        """Simple verification.  To be added to by subclasses."""
        assert isinstance(self.header, self._header_class)
        assert isinstance(self.payload, self._payload_class)
        assert self.payloadsize // 4 == self.payload.words.size

    @property
    def valid(self):
        """Whether frame contains valid data. Can be overridden by subclass."""
        return self._valid

    @valid.setter
    def valid(self, valid):
        self._valid = valid

    @classmethod
    def frombytes(cls, raw, *args, **kwargs):
        """Read a frame set from a byte string.

        Implemented via ``fromfile`` using BytesIO.  For reading from files,
        use ``fromfile`` directly.
        """
        return cls.fromfile(io.BytesIO(raw), *args, **kwargs)

    def tobytes(self):
        return self.header.tobytes() + self.payload.tobytes()

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        valid = kwargs.pop('valid', True)
        verify = kwargs.pop('verify', True)
        header = cls._header_class.fromfile(fh, verify=verify)
        payload = cls._payload_class.fromfile(fh, *args, **kwargs)
        return cls(header, payload, valid=valid, verify=verify)

    def tofile(self, fh):
        return fh.write(self.tobytes())

    @classmethod
    def fromdata(cls, data, header, *args, **kwargs):
        """Construct frame from data and header.

        Parameters
        ----------
        data : ndarray
            Array holding data to be encoded.
        header : VLBIHeaderBase
            Header for the frame.

        *args, **kwargs : arguments
            Additional arguments to help create the payload.

        unless kwargs['verify'] = False, basic assertions that check the
        integrity are made (e.g., that channel information and whether or not
        data are complex are consistent between header and data).

        Returns
        -------
        frame : VLBIFrameBase instance.
        """
        valid = kwargs.pop('valid', True)
        verify = kwargs.pop('verify', True)
        payload = cls._payload_class.fromdata(data, *args, **kwargs)
        return cls(header, payload, valid=valid, verify=verify)

    def todata(self, data=None):
        return self.payload.todata(data)

    data = property(todata, doc="Decode the payload")

    @property
    def shape(self):
        return self.payload.shape

    @property
    def dtype(self):
        return self.payload.dtype

    @property
    def words(self):
        return np.hstack((np.array(self.header.words), self.payload.words))

    @property
    def size(self):
        return self.header.size + self.payload.size

    def __array__(self):
        return self.payload.data

    def __getitem__(self, item):
        # Header behaves as a dictionary.
        return self.header.__getitem__(item)

    def keys(self):
        return self.header.keys()

    def __contains__(self, key):
        return key in self.header.keys()

    def __getattr__(self, attr):
        try:
            return self.__getattribute__(attr)
        except AttributeError:
            if attr in self.header._properties:
                return getattr(self.header, attr)
            else:
                raise

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.valid == other.valid and
                self.header == other.header and
                self.payload == other.payload)
