import pydub
import numpy as np

class Audio:
    def __init__(self, path, file_type):
        self.path = path
        self.file_type = file_type

    def __enter__(self):
        if(self.file_type == "ogg"):
            self.data = pydub.AudioSegment.from_ogg(self.path)
            self.data = self.data.overlay(pydub.AudioSegment.from_file(self.path))
            self.raw = (np.fromstring(self.data._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)  # convert to float

        return (self)

    def __exit__(self, exception_type, exception_value, traceback):
        del self.data
        del self.raw