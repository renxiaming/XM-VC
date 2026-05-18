import os
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import librosa
import numpy as np
from functools import partial

def save_audio(filename, in_dir, out_dir):
    wav, _ = librosa.load(os.path.join(in_dir, filename), sr=48000)
    np.save(os.path.join(out_dir, filename.split('.')[0]), wav)

def save_audios(in_dir, out_dir, generator):
    executor = ProcessPoolExecutor(max_workers=4)
    results = []
    for wav_filename in generator:
        results.append(
            executor.submit(
                partial(save_audio, wav_filename, in_dir, out_dir)
            )
        )
    return [result.result() for result in tqdm(results)]

in_dir = sys.argv[1]
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

if len(sys.argv) >= 4:
    filelist = sys.argv[3]
    filelist = open(filelist).read().splitlines()
    generator = filelist
else:
    generator = os.listdir(in_dir)
print("total", len(generator), "utterances")
save_audios(in_dir, out_dir, generator)