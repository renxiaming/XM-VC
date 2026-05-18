import sys
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Process
import torch
import argparse
import librosa
from aslp_utils import LanceWriter, LanceReader, AudioData, FloatNPYData

torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("in_dir", type=str, help="wav dir")
parser.add_argument("out_dir", type=str, help="out bn feature dir")
parser.add_argument("--filelist", type=str, help="filelst", default=None)
parser.add_argument("--use_lance", type=bool, default=False)

# data_root = sys.argv[1]
# out_data_root = sys.argv[2]
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir
filelist = args.filelist
use_lance = args.use_lance

# in_dir = sys.argv[1]
# out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

def extract_xvector(wav_path, save_path, encoder):
    wav = preprocess_wav(wav_path)
    embed = encoder.embed_utterance(wav)
    np.save(save_path, embed)

def extract_xvector_lance(wav_data, encoder):
    wav = wav_data.audio
    wav = wav / np.iinfo(wav.dtype).max

    if wav_data.sample_rate != 16000:
        if wav_data.sample_rate < 16000:
            print(f"warning: Resample from {wav_data.sample_rate} to 16000")
        wav = librosa.core.resample(
            wav, 
            orig_sr=wav_data.sample_rate,
            target_sr=16000, 
            res_type='kaiser_best')

    embed = encoder.embed_utterance(wav)
    embed = FloatNPYData(wav_data.data_id, data=embed)
    return embed

def extract_xvectors(in_dir, out_dir, wav_filenames):
    encoder = VoiceEncoder()
    wav_filenames = tqdm(wav_filenames)
    for wav_filename in wav_filenames:
        wav_path = os.path.join(in_dir, wav_filename)
        save_path = os.path.join(out_dir, wav_filename.split('.')[0])
        extract_xvector(wav_path, save_path, encoder)
        
if not use_lance:

    if filelist is not None:
        # filelist = sys.argv[3]
        filelist = open(filelist).read().splitlines()
        generator = filelist
    else:
        generator = os.listdir(in_dir)
        
    print("total", len(generator), "utterances")
    extract_xvectors(in_dir, out_dir, generator)
    
else:
    WRITE_INTERVAL = 10000
    
    wav_reader = LanceReader(in_dir, target_cls=AudioData)
    writer = LanceWriter(out_dir, target_cls=FloatNPYData)
    
    wav_ids = wav_reader.get_ids()
    rows = list(range(len(wav_ids)))

    encoder = VoiceEncoder()

    embed_data = []

    for row in tqdm(rows):
        wav_data = wav_reader.get_datas_by_rows([row])[0]
        emded = extract_xvector_lance(wav_data, encoder)
        embed_data.append(emded)
        if len(embed_data) > WRITE_INTERVAL:
            writer.write_parallel(embed_data)
            embed_data = []
        
        # with Pool(self.n_jobs) as pool:
        #     for mel in tqdm(
        #         pool.imap_unordered(mel_function, rows),
        #         total= len(rows)
        #     ):
        #         mel_data.append(mel)
        #         if len(mel_data) > self.WRITE_INTERVAL:
        #             mel_writer.write_parallel(mel_data, progress=True)
        #             mel_data = []
    writer.write_parallel(embed_data)