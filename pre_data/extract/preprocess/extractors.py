# Copyright 2020 The NWPU-ASLP VC team. All Rights Reserved.
#
# Filename     : feature_extractor.py
# Created time : 2020-11-16 10:11
# Last modified: 2020-11-16 10:11
# ==============================================================================
"""Classes for checking labels, loading labels, and extract acoustic features"""

import os
import numpy as np
import pyworld as pw
from tqdm import tqdm
import audio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy import signal
from config import BasicConfig
from aslp_utils import LanceWriter, LanceReader, AudioData, FloatNPYData
import librosa

class BaseAcousticExtractor():
    """
    Base class of acoustic extractor.

    For each acoustic feature extractor, you have to implement the fuction 'extract_features',
    and if you have some special requirements, you may need to reload the __call__ function.

    Noted that the return value (also the orders) is very important for later feature parser.
    """
    def __init__(self, config, n_jobs=20, incremental=False):
        super(BaseAcousticExtractor, self).__init__()
        self.config = config
        self.incremental = incremental
        self.executor = ProcessPoolExecutor(max_workers=n_jobs)

    def extract_features(self, wav_dir, wav_list, out_dir):
        raise NotImplementedError(
            "You need to implement extract_features function based on your feature.")

    def __call__(self, wav_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        wav_list = [os.path.splitext(os.path.basename(filename))[0] for filename in os.listdir(wav_dir) if filename.endswith(".wav")]
        print(f'Start to extract to {out_dir}.')
        print(f"Total count: {len(wav_list)}")
        acoustic_metadata = self.extract_features(wav_dir, wav_list, out_dir)
        return acoustic_metadata


class MelExtractor(BaseAcousticExtractor):
    """
    Mel-spectrogram extractor.

    A spectific mel feature extractor for extracting mel-spectrogram, and it also acts as
    an example for your own extractor.
    """
    def __init__(self, config, n_jobs=20, incremental=False):
        super().__init__(config, n_jobs, incremental)

    def extract_features(self, wav_dir, wav_list, out_dir):
        mel_out_dir = out_dir
        os.makedirs(mel_out_dir, exist_ok=True)
        futures = [self.executor.submit(partial(extract_mel, wav_dir, mel_out_dir, key, self.config)) for key in tqdm(wav_list, desc="Create Process List: ")]
        return [future.result() for future in tqdm(futures) if future.result()]

class MelLanceExtractor(BaseAcousticExtractor):
    """
    Mel-spectrogram extractor.

    A spectific mel feature extractor for extracting mel-spectrogram, and it 
    orgnizes data to lance format.
    """
    def __init__(self, config, n_jobs=20, incremental=False, write_interval = 10000):
        super().__init__(config, n_jobs, incremental)
        self.WRITE_INTERVAL = write_interval
        self.n_jobs = n_jobs

    # def extract_features(self, wav_dir, wav_list, out_dir):
    #     mel_out_dir = out_dir
    #     os.makedirs(mel_out_dir, exist_ok=True)
    #     futures = [self.executor.submit(partial(extract_mel, wav_dir, mel_out_dir, key, self.config)) for key in tqdm(wav_list, desc="Create Process List: ")]
    #     return [future.result() for future in tqdm(futures) if future.result()]
    
    def __call__(self, wav_lance_dir, out_lance_dir, filelist=None):
        wav_reader = LanceReader(wav_lance_dir, target_cls=AudioData)
        wav_ids = wav_reader.get_ids()
        mel_writer = LanceWriter(out_lance_dir, target_cls=FloatNPYData)
        
        from multiprocessing import Pool
        
        mel_function = partial(extract_mel_lance, wav_reader=wav_reader, hparams = self.config)
        if filelist is not None:
            file_set = set()
            for line in open(filelist, 'r').readlines():
                file_set.add(line.strip())

            rowids = [wav_id._rowid for i, wav_id in enumerate(wav_ids) if wav_id.data_id in file_set]
        else:
            # rows = list(range(len(wav_ids)))
            print("-------wav_ids------",len(wav_ids))
            processed_ids = mel_writer.get_ids()
            print("-------processed_ids------",len(processed_ids))
            processed_ids = set([id.data_id for id in processed_ids])
            rowids = [wav_id._rowid for wav_id in wav_ids if wav_id.data_id not in processed_ids]
            # rowids = [wav_id._rowid for wav_id in wav_ids]
        
        mel_data = []
        
        # import pdb;pdb.set_trace()
        
        for row in tqdm(rowids):
            mel = mel_function(row)
            mel_data.append(mel)
            if len(mel_data) > self.WRITE_INTERVAL:
                mel_writer.write_parallel(mel_data)
                mel_data = []
        
        # with Pool(self.n_jobs) as pool:
        #     for mel in tqdm(
        #         pool.imap_unordered(mel_function, rows),
        #         total= len(rows)
        #     ):
        #         mel_data.append(mel)
        #         if len(mel_data) > self.WRITE_INTERVAL:
        #             mel_writer.write_parallel(mel_data, progress=True)
        #             mel_data = []
        mel_writer.write_parallel(mel_data)
        
        # os.makedirs(out_dir, exist_ok=True)
        # wav_list = [os.path.splitext(os.path.basename(filename))[0] for filename in os.listdir(wav_dir) if filename.endswith(".wav")]
        # print(f'Start to extract to {out_dir}.')
        # print(f"Total count: {len(wav_list)}")
        # acoustic_metadata = self.extract_features(wav_dir, wav_list, out_dir)
        # return acoustic_metadata


class LinearExtractor(BaseAcousticExtractor):
    """
    Mel-spectrogram extractor.

    A spectific mel feature extractor for extracting mel-spectrogram, and it also acts as
    an example for your own extractor.
    """
    def __init__(self, config, n_jobs=20, incremental=False):
        super().__init__(config, n_jobs, incremental)

    def extract_features(self, wav_dir, wav_list, out_dir):
        linear_out_dir = os.path.join(out_dir, "linear_spec")
        os.makedirs(linear_out_dir, exist_ok=True)
        futures = [self.executor.submit(partial(extract_linear, wav_dir, linear_out_dir, key, self.config)) for key in tqdm(wav_list, desc="Create Process List: ")]
        return [future.result() for future in tqdm(futures, desc="Extracting: ") if future.result()]

class Lf0Extractor(BaseAcousticExtractor):
    """
    Lf0  extractor.

    A spectific mel feature extractor for extracting mel-spectrogram,lf0,uv and energy, and it also acts as
    an example for your own extractor.
    """
    def __init__(self, config, n_jobs=20, incremental=False):
        super().__init__(config, n_jobs, incremental)

    def extract_features(self, wav_dir, wav_list, out_dir):
        lf0_out_dir = os.path.join(out_dir, "lf0")
        os.makedirs(lf0_out_dir, exist_ok=True)
        futures = [self.executor.submit(partial(extract_lf0, wav_dir, lf0_out_dir, key, self.config)) for key in tqdm(wav_list, desc="Create Process List: ")]
        return [future.result() for future in tqdm(futures, desc="Extracting: ") if future.result()]

class EnergyExtractor(BaseAcousticExtractor):
    """
    Lf0  extractor.

    A spectific mel feature extractor for extracting mel-spectrogram,lf0,uv and energy, and it also acts as
    an example for your own extractor.
    """
    def __init__(self, config, n_jobs=20, incremental=False):
        super().__init__(config, n_jobs, incremental)

    def extract_features(self, wav_dir, wav_list, out_dir):
        energy_out_dir = os.path.join(out_dir, "energys")
        os.makedirs(energy_out_dir, exist_ok=True)
        futures = [self.executor.submit(partial(extract_energy, wav_dir, energy_out_dir, key, self.config)) for key in tqdm(wav_list, desc="Create Process List: ")]
        return [future.result() for future in tqdm(futures, desc="Extracting: ") if future.result()]


class MelLf0UVEnergyExtractor(BaseAcousticExtractor):
    """
    Mel Lf0 uv energy  extractor.

    A spectific mel feature extractor for extracting mel-spectrogram,lf0,uv and energy, and it also acts as
    an example for your own extractor.
    """
    def __init__(self, config, n_jobs=20, incremental=False):
        super().__init__(config, n_jobs, incremental)

    def extract_features(self, wav_dir, wav_list, out_dir):
        mel_out_dir = os.path.join(out_dir, "mels")
        lf0_out_dir = os.path.join(out_dir, "lf0")
        uv_out_dir = os.path.join(out_dir, "uv")
        energy_out_dir = os.path.join(out_dir, "energys")
        os.makedirs(mel_out_dir, exist_ok=True)
        os.makedirs(lf0_out_dir, exist_ok=True)
        os.makedirs(uv_out_dir, exist_ok=True)
        os.makedirs(energy_out_dir, exist_ok=True)
        futures = [self.executor.submit(partial(extract_mel_lf0_uv_energy, wav_dir, mel_out_dir, lf0_out_dir, uv_out_dir, energy_out_dir, key, self.config)) for key in tqdm(wav_list, desc="Create Process List: ")]
        return [future.result() for future in tqdm(futures, desc="Extracting: ") if future.result()]
    
def process_wav(wav_path, hparams):
    wav = audio.load_wav(
        wav_path,
        target_sr=hparams.sample_rate,
        win_size=hparams.win_size,
        hop_size=hparams.hop_size)
    # Process wav samples
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    if hparams.trim_silence:
        wav = audio.trim_silence(wav, hparams)
    return wav

def extract_mel(wav_dir, out_dir, key, hparams):
    wav = process_wav(os.path.join(wav_dir, key+".wav"), hparams)
    mel, _, _ = __extract_mel(wav, hparams)
    np.save(os.path.join(out_dir, key + ".npy"), mel)
    
    
def load_lance_wav(wav_data:AudioData, hparams):
    wav = wav_data.audio
    wav = wav / np.iinfo(wav.dtype).max
    
    if wav_data.sample_rate != hparams.sample_rate:
        if wav_data.sample_rate < hparams.sample_rate:
            print(f"warning: Resample from {wav_data.sample_rate} to {hparams.sample_rate}")
        wav = librosa.core.resample(
            wav, 
            orig_sr=wav_data.sample_rate,
            target_sr=hparams.sample_rate, 
            res_type='kaiser_best')
    
    target_length = (wav.size // hparams.hop_size) * hparams.hop_size # + win_size // hop_size) * hop_size
    pad_len = (target_length - wav.size) // 2
    if pad_len < 0:
        wav = wav[:target_length]
    else:
        if wav.size % 2 == 0:
            wav = np.pad(wav, (pad_len, pad_len), mode='reflect')
        else:
            wav = np.pad(wav, (pad_len, pad_len + 1), mode='reflect')
    
    if hparams.trim_silence:
        wav = audio.trim_silence(wav, hparams)
            
    return wav
    

def extract_mel_lance(rowid, wav_reader: LanceReader, hparams):
    wav_data = wav_reader.get_datas_by_rowids([rowid])[0]
    
    # import pdb;pdb.set_trace()
    
    # tmp = audio.load_wav(
    #     '/home/work_nfs16/zhguo/code/dualvc/testdata/wavs/SSB00730313.wav',
    #     target_sr=hparams.sample_rate,
    #     win_size=hparams.win_size,
    #     hop_size=hparams.hop_size)
    
    wav = load_lance_wav(wav_data, hparams)
        
    mel, _, _ = __extract_mel(wav, hparams)
    return FloatNPYData(wav_data.data_id, data=mel)


def extract_linear(wav_dir, out_dir, key, hparams):
    wav = process_wav(os.path.join(wav_dir, key+".wav"), hparams)
    linear_spec, _, _ = __extract_linear(wav, hparams)
    np.save(os.path.join(out_dir, key + ".npy"), linear_spec)

def extract_lf0(wav_dir, out_dir, key, hparams):
    wav = process_wav(os.path.join(wav_dir, key+".wav"), hparams)
    lf0, uv, _, _ = __extract_lf0(wav, hparams)
    np.save(os.path.join(out_dir, key + ".npy"), lf0.astype(np.float32))

def extract_energy(wav_dir, out_dir, key, hparams):
    wav = process_wav(os.path.join(wav_dir, key+".wav"), hparams)
    energy, _, _ = __extract_energy(wav, hparams)
    np.save(os.path.join(out_dir, key + ".npy"), energy.astype(np.float32))

def extract_mel_lf0_uv_energy(wav_dir, mel_out_dir, lf0_out_dir, uv_out_dir, energy_out_dir, key, hparams):
    wav = process_wav(os.path.join(wav_dir, key+".wav"), hparams)
    mel_filename = os.path.join(mel_out_dir, key + '.npy')
    energy_filename = os.path.join(energy_out_dir, key + '.npy')
    lf0_filename = os.path.join(lf0_out_dir, key + '.npy')
    uv_filename = os.path.join(uv_out_dir, key + '.npy')

    # Extract mel spectrogram
    mel_spectrogram, _, _ = __extract_mel(wav, hparams)
    np.save(mel_filename, mel_spectrogram, allow_pickle=False)

    lf0, uv, _, _ = __extract_lf0(wav, hparams)
    np.save(lf0_filename, lf0.astype(np.float32), allow_pickle=False)
    np.save(uv_filename, uv.astype(np.float32), allow_pickle=False)

    energy, _, _ = __extract_energy(wav, hparams)
    np.save(energy_filename, energy.astype(np.float32), allow_pickle=False)

def __extract_mel(wav, hparams):
    n_samples = len(wav)

    # Extract mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    n_frames = mel_spectrogram.shape[1]


    # Align features
    desired_frames = int(min(n_samples / hparams.hop_size, n_frames))
    wav = wav[:desired_frames * hparams.hop_size]
    mel_spectrogram = mel_spectrogram[:, :desired_frames]
    n_samples = wav.shape[0]
    n_frames = mel_spectrogram.shape[1]
    assert(n_samples / hparams.hop_size == n_frames)

    return (mel_spectrogram, n_samples, n_frames)

def __extract_linear(wav, hparams):
    n_samples = len(wav)

    # Extract linear spectrogram
    linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
    n_frames = linear_spectrogram.shape[1]

    # Align features
    desired_frames = int(min(n_samples / hparams.hop_size, n_frames))
    wav = wav[:desired_frames * hparams.hop_size]
    linear_spectrogram = linear_spectrogram[:, :desired_frames]
    n_samples = wav.shape[0]
    n_frames = linear_spectrogram.shape[1]
    assert(n_samples / hparams.hop_size == n_frames)

    return (linear_spectrogram, n_samples, n_frames)

def interpolate_f0(data):
    data = np.reshape(data, (data.size, 1))
    vuv_vector = np.zeros((data.size, 1),dtype=np.float32)
    vuv_vector[data > 0.0] = 1.0
    vuv_vector[data <= 0.0] = 0.0
    ip_data = data
    frame_number = data.size
    last_value = 0.0
    for i in range(frame_number):
        if data[i] <= 0.0:
            j = i + 1
            for j in range(i + 1, frame_number):
                if data[j] > 0.0:
                    break
            if j < frame_number - 1:
                if last_value > 0.0:
                    step = (data[j] - data[i - 1]) / float(j - i)
                    for k in range(i, j):
                        ip_data[k] = data[i - 1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        ip_data[k] = data[j]
            else:
                for k in range(i, frame_number):
                    ip_data[k] = last_value
        else:
            ip_data[i] = data[i]
            last_value = data[i]
    return ip_data, vuv_vector

def __extract_lf0(wav, hparams):
    n_samples = len(wav)
    
    #extract lf0
    sound = wav.astype(np.double)
    f0, sp, ap = pw.wav2world(sound, hparams.sample_rate, frame_period=hparams.hop_size/hparams.sample_rate*1000)
    lf0 = np.log(f0 + 1e-8)
    lf0[lf0 < 1e-3] = 0
    lf0 = lf0.astype(np.float32)

    if hparams.lf0_inter:
        lf0_data, uv = interpolate_f0(lf0)
    else:
        lf0_data, uv = interpolate_f0(lf0.copy())
    if hparams.lf0_norm == 'minmax':
        lf0_data = (lf0_data - min(lf0_data))/(max(lf0_data) - min(lf0_data) + 1e-6)
    elif hparams.lf0_norm == 'z-score':
        mean = np.mean(lf0_data[uv > 0.0])
        std = np.std(lf0_data[uv > 0.0])
        lf0_data = (lf0_data - mean)/std * uv
        lf0_data[uv <= 0.0] = -4.0
    elif hparams.lf0_norm == 'raw':
        lf0_data = lf0
    else:
        raise Exception('{} does\'t support now!'.format(hparams.lf0_norm))
    n_frames = lf0_data.shape[0]

    return (lf0_data, uv, n_samples, n_frames)

def __extract_energy(wav, hparams):
    n_samples = len(wav)
    energys = []
    for i in range(0, len(wav) - hparams.win_size, hparams.hop_size):
        power = np.sum(np.absolute(wav[i: i + hparams.win_size])) / hparams.win_size
        power = np.clip(power, 1e-9, 1e9)
        energys.append([np.log10(power)])
    energy = np.array(energys)
    if hparams.energy_norm == 'minmax':
        energy = (energy - (-4.5))/(max(energy) - (-4.5) + 1e-6)
        #energy = (energy - min(energy)) / (max(energy) - min(energy) + 1e-6)
    elif hparams.energy_norm == 'z-score':
        mean = np.nanmean(energy)
        std = np.nanstd(energy)
        energy = (energy - mean) / std * uv
    elif hparams.energy_norm == 'raw':
        energy = energy
    else:
        raise Exception('{} does\'t support now!'.format(hparams.energy_norm))
    n_frames = energy.shape[0]
    return energy, n_samples, n_frames
