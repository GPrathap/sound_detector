"""
PyAudio example: Record a few seconds of audio and save to a WAVE
file.
"""

import pyaudio
import wave
import sys
import numpy as np

CHUNK = 44100
FORMAT = pyaudio.paInt32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 120

WAVE_OUTPUT_FILENAME = "output.wav"
WAVE_OUTPUT_FILENAME1 = "output1.wav"
WAVE_OUTPUT_FILENAME2 = "output2.wav"

if sys.platform == 'darwin':
    CHANNELS = 1

p = pyaudio.PyAudio()

p1 = pyaudio.PyAudio()

p2 = pyaudio.PyAudio()


stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK, input_device_index=2)
stream1 = p1.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK, input_device_index=3)
stream2 = p2.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK, input_device_index=4)



print("* recording")

frames = []
frames1 = []
frames2 = []


for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK, exception_on_overflow = False)
    data1 = stream1.read(CHUNK, exception_on_overflow = False)
    data2 = stream2.read(CHUNK, exception_on_overflow = False)

    
    
    #print data
    try:
       #print "--------1"
       frames.append(data)
       frames1.append(data1)
       frames2.append(data2)


      # print "---->>>"
       samps = np.fromstring(data, dtype=np.int16)
       samps1 = np.fromstring(data1, dtype=np.int16)
       samps2 = np.fromstring(data2, dtype=np.int16)
       #print "--------2"
       value = np.sum(np.absolute(samps))
       value1 = np.sum(np.absolute(samps1))
       value2 = np.sum(np.absolute(samps2))
       #print samps
       #print value
       #print value1
       #print value2

    except:
       print("Unexpected error:", sys.exc_info()[0])
       #frames.append(data)
       pass 

print("* done recording")

stream.stop_stream()
stream1.stop_stream()
stream2.stop_stream()

stream.close()
stream1.close()
stream2.close()
p.terminate()
p1.terminate()
p2.terminate()



wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
wf1.setnchannels(CHANNELS)
wf1.setsampwidth(p.get_sample_size(FORMAT))
wf1.setframerate(RATE)
wf1.writeframes(b''.join(frames))
wf1.close()


wf2 = wave.open(WAVE_OUTPUT_FILENAME2, 'wb')
wf2.setnchannels(CHANNELS)
wf2.setsampwidth(p.get_sample_size(FORMAT))
wf2.setframerate(RATE)
wf2.writeframes(b''.join(frames))
wf2.close()

