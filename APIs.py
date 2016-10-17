import numpy as np
import pydub
import librosa
import os
import shutil
import urllib
import zipfile
import glob
import matplotlib
import matplotlib.pyplot as plt
from operator import add

from sys import getsizeof

import tensorflow as tf
import numpy as np
from operator import add

import seaborn as sb
import time

sb.set(style="white", palette="muted")

import pandas as pd
import random
random.seed(20150420)
numOfClasses = 10

class Clip:
    """A single 5-sec long recording."""

    RATE = 44100  # All recordings in ESC are 44.1 kHz
    FRAME = 1024  # Frame size in samples

    class Audio:
        """The actual audio data of the clip.

            Uses a context manager to load/unload the raw audio data. This way clips
            can be processed sequentially with reasonable memory usage.
        """

        def __init__(self, path):
            self.path = path

        def __enter__(self):
            # Actual recordings are sometimes not frame accurate, so we trim/overlay to exactly 5 seconds
            self.data = pydub.AudioSegment.silent(duration=5000)
            self.data = self.data.overlay(pydub.AudioSegment.from_file(self.path)[0:5000])
            self.raw = (np.fromstring(self.data._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)  # convert to float
            return (self)

        def __exit__(self, exception_type, exception_value, traceback):
            #if exception_type is not None:
                #print exception_type, exception_value, traceback
            del self.data
            del self.raw

    def __init__(self, filename):
        self.filename = os.path.basename(filename)
        self.path = os.path.abspath(filename)
        self.directory = os.path.dirname(self.path)
        self.category = self.directory.split('/')[-1]

        self.audio = Clip.Audio(self.path)

        with self.audio as audio:
            self._compute_mfcc(audio)
            self._compute_zcr(audio)
            self._compute_fft(audio)

    def _compute_mfcc(self, audio):
        # MFCC computation with default settings (2048 FFT window length, 512 hop length, 128 bands)
        self.melspectrogram = librosa.feature.melspectrogram(audio.raw, sr=Clip.RATE, hop_length=Clip.FRAME)
        self.logamplitude = librosa.logamplitude(self.melspectrogram)
        self.mfcc = librosa.feature.mfcc(S=self.logamplitude, n_mfcc=256).transpose()

    def _compute_fft(self, audio):
        self.fft = []
        frames = int(np.ceil(len(audio.data) / 1000.0 * Clip.RATE / Clip.FRAME))

        for i in range(0, frames):
            frame = Clip._get_frame(audio, i)
            ps = np.fft.fft(frame)
            self.fft.append(np.abs(ps))

        self.fft = np.asarray(self.fft)



    def _compute_zcr(self, audio):
        # Zero-crossing rate
        self.zcr = []
        frames = int(np.ceil(len(audio.data) / 1000.0 * Clip.RATE / Clip.FRAME))

        for i in range(0, frames):
            frame = Clip._get_frame(audio, i)
            self.zcr.append(np.mean(0.5 * np.abs(np.diff(np.sign(frame)))))

        self.zcr = np.asarray(self.zcr)

    @classmethod
    def _get_frame(cls, audio, index):
        if index < 0:
            return None
        return audio.raw[(index * Clip.FRAME):(index + 1) * Clip.FRAME]

    def __repr__(self):
        return '<{0}/{1}>'.format(self.category, self.filename)


def getClassArray():
    return [0] * numOfClasses


def load_dataset(name):
    """Load all dataset recordings into a nested list."""
    clips = []

    datasetXForConvolution = []
    datasetYForConvolution = []
    datasetXForFull = []
    datasetYForFull = []
    for directory in sorted(os.listdir('{0}/'.format(name))):
        directory = '{0}/{1}'.format(name, directory)
        if os.path.isdir(directory) and os.path.basename(directory)[0:3].isdigit():
            print('Parsing ' + directory)
            category = []
            for clip in sorted(os.listdir(directory)):
                if clip[-3:] == 'ogg':
                    audioFile = Clip('{0}/{1}'.format(directory, clip))
                    numberOfWindows = len(audioFile.mfcc)

                    featureSetPerWindowXForConvolution = audioFile.mfcc
                    datasetXForConvolution.append(featureSetPerWindowXForConvolution.tolist())

                    featureSetPerWindowXForFull = audioFile.zcr
                    datasetXForFull.append(featureSetPerWindowXForFull.tolist())

                    for i in range(0, numberOfWindows):
                        classes = getClassArray()
                        classes[int('{1}'.format(name, directory).split("/")[1].split("-")[0]) - 1] = 1
                        datasetYForConvolution.append(classes)
                        ##after adding new features add it to either one
                        datasetYForFull.append(classes)

                    category.append(audioFile)
            clips.append(category)

    print('All {0} recordings loaded.'.format(name))
    return clips , datasetXForConvolution[0], datasetYForConvolution , datasetXForFull, datasetYForFull









































