from functools import partial
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Process
import torch
import random
import glob
import librosa
import soundfile
from scipy.io.wavfile import write
from aslp_utils import LanceWriter, LanceReader, AudioData
from multiprocessing import Pool
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

torch.set_num_threads(1)

in_dir = sys.argv[1]
out_dir = sys.argv[2]
use_lance = bool(sys.argv[3])

sr=16000

class RandomNoise:

    def __init__(self):
        # self.noise_list = glob.glob("/project/tts/tts_expdata/nzq/musan/musan/noise_48/*.wav")
        # self.speech_list = glob.glob("/project/tts/tts_expdata/nzq/musan/musan/speech_48/*.wav")
        self.noise_list = [line.strip() for line in open('/home/node8_tmpdata/zhguo/data/noise/esc16k.scp', 'r').readlines()]
        self.noise_db_low = 20.
        self.noise_db_high = 25.

    def _add_noise(self, speech, power):
        nsamples = speech.shape[0]
        # if random.randint(0, 1) == 0:
        #     noise_path = np.random.choice(self.noise_list)
        # else:
        #     noise_path = np.random.choice(self.speech_list)
        noise_path = np.random.choice(self.noise_list)
        noise = None
        noise_db = np.random.uniform(self.noise_db_low, self.noise_db_high)
        #print("loading noise")
        noise, _ = librosa.load(noise_path, sr=sr)
        #print("loaded noise")
        new_noise = noise
        #print("before", speech.shape, noise.shape)
        while new_noise.shape[0] <= nsamples:
            new_noise = np.concatenate([new_noise, noise], axis=0)
        noise = new_noise
        offset = np.random.randint(0, noise.shape[0] - speech.shape[0])
        #print("after", speech.shape, noise.shape)
        # noise: (Nmic, Time)
        noise = noise[offset:nsamples+offset]

        noise_power = (noise**2).mean()
        scale = (
            np.sqrt(power)
            / np.sqrt(max(noise_power, 1e-10))
        )
        speech = speech + scale * noise
        return speech

    def add_noises(self, in_dir, out_dir, wav_filenames):
        for wav_filename in tqdm(wav_filenames):
            #print("loading wav")
            wav, _ = librosa.load(os.path.join(in_dir, wav_filename), sr=sr)
            #print("loaded wav")
            #wav = np.expand_dims(wav, axis=0)
            power = (wav ** 2).mean()
            wav_noise = self._add_noise(wav, power)
            write(os.path.join(out_dir, wav_filename), rate=sr, data=wav_noise)
        

noise_generator = RandomNoise()

def add_noise_lance(row, reader: LanceReader):
    import pdb;pdb.set_trace()
    data = reader.get_datas_by_rows([row])[0]
    in_wav = data.audio
    in_wav = in_wav / np.iinfo(in_wav.dtype).max
    noised_speech = noise_generator._add_noise(in_wav, (in_wav ** 2).mean())
    noised_speech = (noised_speech * np.iinfo(np.int16).max).astype(np.int16)
    return AudioData(data.data_id+'_noise', audio = noised_speech, sample_rate=data.sample_rate, duration=data.duration)


# if len(sys.argv) >= 4:
#     filelist = sys.argv[3]
#     filelist = open(filelist).read().splitlines()
#     generator = filelist
# else:
#     generator = os.listdir(in_dir)
# print("total", len(generator), "utterances")
# noise_generator.add_noises(in_dir, out_dir, generator)


if __name__ == "__main__":


    if not use_lance:
        
        
        assert NotImplementedError

    else:
        
        WRITE_INTERVAL = 10000
        
        wav_reader = LanceReader(in_dir, target_cls=AudioData)
        writer = LanceWriter(out_dir, target_cls=AudioData)
        
        print("readling data ids")
        wav_ids = wav_reader.get_ids()
        rows = list(range(len(wav_ids)))
        print("data ids loaded")
        
        wav_data = []

        # import pdb;pdb.set_trace()
        tempo_function = partial(add_noise_lance, reader=wav_reader)

        # for row in rows:
        #     tempo_function(row)

        with Pool(10) as pool:
            for i in tqdm(
                pool.imap_unordered(tempo_function, rows),
                total=len(rows)
            ):
                if i == None:
                    continue
                wav_data.append(i)
                if len(wav_data) > WRITE_INTERVAL:
                    writer.write_parallel(wav_data)
                    wav_data = []
        writer.write_parallel(wav_data)
        