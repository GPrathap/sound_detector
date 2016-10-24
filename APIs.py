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
numberOfBins = 32
numberOfMfcc = 16
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
            """ Feature used for convolutional neural network """
            self._compute_mfcc(audio)
            self._compute_fft(audio)
            """ Time-domain audio features for full layer network """
            #self._compute_zcr(audio)
            #self._compute_energy(audio)
            #self._compute_energy_entropy(audio)
            """ Frequency-domain audio features for full layer network """
            #add if you have more features



    def _compute_mfcc(self, audio):
        # MFCC computation with default settings (2048 FFT window length, 512 hop length, 128 bands)
        self.melspectrogram = librosa.feature.melspectrogram(audio.raw, sr=Clip.RATE, hop_length=Clip.FRAME)
        self.logamplitude = librosa.logamplitude(self.melspectrogram)
        self.mfcc = librosa.feature.mfcc(S=self.logamplitude, n_mfcc=numberOfMfcc).transpose()
        self.mfcc_delta = librosa.feature.delta(self.mfcc)
        self.mfcc_delta2 = librosa.feature.delta(self.mfcc, order=2)
        self.logamplitude = self.logamplitude.T.flatten()[:, np.newaxis].T

    def _compute_fft(self, audio):
        self.fft = []
        frames = int(np.ceil(len(audio.data) / 1000.0 * Clip.RATE / Clip.FRAME))
        for i in range(0, frames):
            frame = Clip._get_frame(audio, i)
            ps = np.fft.fft(frame, numberOfBins)
            self.fft.append(librosa.logamplitude(np.abs(ps)**2))
        self.fft = np.asarray(self.fft)

    def _compute_energy(self, audio):
        # Computes signal energy of frame
        self.energy = []
        frames = int(np.ceil(len(audio.data) / 1000.0 * Clip.RATE / Clip.FRAME))
        for i in range(0, frames):
            frame = Clip._get_frame(audio, i)
            self.energy.append(np.sum(frame ** 2) / np.float64(len(frame)))
        self.energy = np.asarray(self.energy)

    def _compute_energy_entropy(self, audio):
        #  Computes entropy of energy
        numOfShortBlocks = 10
        eps = 0.00000001
        self.energy_entropy = []
        frames = int(np.ceil(len(audio.data) / 1000.0 * Clip.RATE / Clip.FRAME))
        for i in range(0, frames):
            frame = Clip._get_frame(audio, i)
            Eol = np.sum(frame ** 2)  # total frame energy
            L = len(frame)
            subWinLength = int(np.floor(L / numOfShortBlocks))
            if L != subWinLength * numOfShortBlocks:
                frame = frame[0:subWinLength * numOfShortBlocks]
            # subWindows is of size [numOfShortBlocks x L]
            subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()
            # Compute normalized sub-frame energies:
            s = np.sum(subWindows ** 2, axis=0) / (Eol + eps)
            # Compute entropy of the normalized sub-frame energies:
            entropy = -np.sum(s * np.log2(s + eps))
            self.energy_entropy.append(entropy)
        self.energy_entropy = np.asarray(self.energy_entropy)



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


def load_dataset_plot(name):
    """Load all dataset recordings into a nested list."""
    clips = []

    for directory in sorted(os.listdir('{0}/'.format(name))):
        directory = '{0}/{1}'.format(name, directory)
        if os.path.isdir(directory) and os.path.basename(directory)[0:3].isdigit():
            print('Parsing ' + directory)
            category = []
            for clip in sorted(os.listdir(directory)):
                if clip[-3:] == 'ogg':
                    category.append(Clip('{0}/{1}'.format(directory, clip)))
            clips.append(category)
    print('All {0} recordings loaded.'.format(name))

    return clips

class ClipRealTime:
    """A single 5-sec long recording."""

    RATE = 44100  # All recordings in ESC are 44.1 kHz
    FRAME = 1024  # Frame size in samples

    class Audio:
        """The actual audio data of the clip.

            Uses a context manager to load/unload the raw audio data. This way clips
            can be processed sequentially with reasonable memory usage.
        """

        def __init__(self, rowData):
            self.rowData = rowData
            self.frames = int(np.ceil(len(self.rowData) / ClipRealTime.FRAME))

        def __enter__(self):
            # Actual recordings are sometimes not frame accurate, so we trim/overlay to exactly 5 seconds
            return (self)

        def __exit__(self, exception_type, exception_value, traceback):
            #if exception_type is not None:
                #print exception_type, exception_value, traceback
            del self.rowData

    def __init__(self, rowData):
        self.audio = ClipRealTime.Audio(rowData)

        with self.audio as audio:
            """ Feature used for convolutional neural network """
            self._compute_mfcc(audio.rowData)
            self._compute_fft(audio.rowData)
            """ Time-domain audio features for full layer network """
            #self._compute_zcr(audio)
            #self._compute_energy(audio)
            #self._compute_energy_entropy(audio)
            """ Frequency-domain audio features for full layer network """
            #add if you have more features

    def _compute_mfcc(self, audio):
        # MFCC computation with default settings (2048 FFT window length, 512 hop length, 128 bands)
        self.melspectrogram = librosa.feature.melspectrogram(audio, sr=ClipRealTime.RATE, hop_length=ClipRealTime.FRAME)
        self.logamplitude = librosa.logamplitude(self.melspectrogram)
        self.mfcc = librosa.feature.mfcc(S=self.logamplitude, n_mfcc=numberOfMfcc).transpose()
        self.mfcc_delta = librosa.feature.delta(self.mfcc)
        self.mfcc_delta2 = librosa.feature.delta(self.mfcc, order=2)
        self.logamplitude = self.logamplitude.T.flatten()[:, np.newaxis].T

    def _compute_fft(self, audio):
        self.fft = []
        frames = self.audio.frames
        for i in range(0, frames):
            frame = ClipRealTime._get_frame(audio, i)
            ps = np.fft.fft(frame, numberOfBins)
            self.fft.append(librosa.logamplitude(np.abs(ps)**2))
        self.fft = np.asarray(self.fft)

    def _compute_energy(self, audio):
        # Computes signal energy of frame
        self.energy = []
        frames = self.audio.frames
        for i in range(0, frames):
            frame = ClipRealTime._get_frame(audio, i)
            self.energy.append(np.sum(frame ** 2) / np.float64(len(frame)))
        self.energy = np.asarray(self.energy)

    def _compute_energy_entropy(self, audio):
        #  Computes entropy of energy
        numOfShortBlocks = 10
        eps = 0.00000001
        self.energy_entropy = []
        frames = self.audio.frames
        for i in range(0, frames):
            frame = Clip._get_frame(audio, i)
            Eol = np.sum(frame ** 2)  # total frame energy
            L = len(frame)
            subWinLength = int(np.floor(L / numOfShortBlocks))
            if L != subWinLength * numOfShortBlocks:
                frame = frame[0:subWinLength * numOfShortBlocks]
            # subWindows is of size [numOfShortBlocks x L]
            subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()
            # Compute normalized sub-frame energies:
            s = np.sum(subWindows ** 2, axis=0) / (Eol + eps)
            # Compute entropy of the normalized sub-frame energies:
            entropy = -np.sum(s * np.log2(s + eps))
            self.energy_entropy.append(entropy)
        self.energy_entropy = np.asarray(self.energy_entropy)



    def _compute_zcr(self, audio):
        # Zero-crossing rate
        self.zcr = []
        frames = audio.frames
        for i in range(0, frames):
            frame = ClipRealTime._get_frame(audio, i)
            self.zcr.append(np.mean(0.5 * np.abs(np.diff(np.sign(frame)))))
        self.zcr = np.asarray(self.zcr)

    @classmethod
    def _get_frame(cls, audio, index):
        if index < 0:
            return None
        return audio[(index * ClipRealTime.FRAME):(index + 1) * ClipRealTime.FRAME]


def getClassArray():
    return [0] * numOfClasses


def load_dataset_realtime(audioData):
    """Load all dataset recordings into a nested list."""
    clips = []

    datasetXForConvolution = []
    datasetYForConvolution = []
    datasetXForFull = []
    datasetYForFull = []
    audioFile = audioData
    numberOfWindows = len(audioFile.mfcc) -1
    mfccFeq=audioFile.mfcc.tolist()
    mfccFeqDelta = audioFile.mfcc_delta.tolist()
    mfccFeqDelta2 = audioFile.mfcc_delta2.tolist()
    fftFeq = audioFile.fft.tolist()
    for i in range(0, numberOfWindows):
        classes = getClassArray()
        classes[0] = 0
        datasetXForConvolution.append(mfccFeq[i]+mfccFeqDelta[i]+ mfccFeqDelta2[i] + fftFeq[i])
        datasetYForConvolution.append(classes)
        #datasetYForFull.append(classes)
        #datasetXForFull.append([audioFile.zcr[i],  audioFile.energy[i], audioFile.energy_entropy[i]]
    print('Calculated all features')
    return clips ,  datasetXForConvolution, datasetYForConvolution , datasetXForFull, datasetYForFull


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
                    mfccFeq=audioFile.mfcc.tolist()
                    mfccFeqDelta = audioFile.mfcc_delta.tolist()
                    mfccFeqDelta2 = audioFile.mfcc_delta2.tolist()
                    fftFeq = audioFile.fft.tolist()
                    for i in range(0, numberOfWindows):
                        classes = getClassArray()
                        classes[int('{1}'.format(name, directory).split("/")[1].split("-")[0]) - 1] = 1
                        datasetXForConvolution.append(mfccFeq[i]+mfccFeqDelta[i]+ mfccFeqDelta2[i] + fftFeq[i])
                        datasetYForConvolution.append(classes)
                        #datasetYForFull.append(classes)
                        #datasetXForFull.append([audioFile.zcr[i],  audioFile.energy[i], audioFile.energy_entropy[i]])
                    category.append(audioFile)
            clips.append(category)

    print('All {0} recordings loaded.'.format(name))
    return clips ,  datasetXForConvolution, datasetYForConvolution , datasetXForFull, datasetYForFull





# for i in range(0,3):
# 	a = np.random.rand(1,40960)[0]
# 	classifiedArray = classifySound.classifySound(a)



































