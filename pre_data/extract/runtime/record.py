import aubio
import numpy as np
import pyaudio
import wave
from tqdm import tqdm

BUFFER_SIZE=2048
CHUNK=200
FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=16000
RECORD_SECONDS=5
WAVE_OUTPUT_FILENAME="test.wav"
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

pDetection = aubio.pitch("default", 1024, 200, 16000)
    # Set unit.
pDetection.set_unit("Hz")
# Frequency under -40 dB will considered
# as a silence.
pDetection.set_silence(-40)

print("* recording")

frames = []

for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS))):
    data = stream.read(CHUNK)
    frames.append(data)

    samples = np.fromstring(data, dtype=aubio.float_type)
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (1 << 15)
    print(samples.shape)
    pitch = pDetection(samples)
    print(pitch)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
