import sys
import os
import librosa
from tqdm import tqdm

sr = 48000
in_dir = sys.argv[1]
out_file = sys.argv[2]

if len(sys.argv) >= 4:
    in_file = sys.argv[3]
    out_file = sys.argv[3] + "_filter"
    generator = open(in_file).read().splitlines()
else:
    generator = os.listdir(in_dir)

good_filenames = ""
for wav_filename in tqdm(generator):
    wav, _ = librosa.load(os.path.join(in_dir, wav_filename), sr=sr)
    if wav.shape[0] < sr * 0.3 or wav.shape[0] > sr * 30:
        continue
    good_filenames = good_filenames + wav_filename + "\n"

sorted(good_filenames)
open(out_file, "w").write(good_filenames)
