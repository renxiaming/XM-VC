# Copyright 2020 The NWPU-ASLP VC team. All Rights Reserved.
#
# Filename     : feature_extractor.py
# Created time : 2020-11-16 10:11
# Last modified: 2020-11-16 10:11
# ==============================================================================
"""Classes for checking labels, loading labels, and extract acoustic features"""

import os
import random
import librosa
import argparse
import numpy as np
import pyworld as pw
from multiprocessing import cpu_count
from tqdm import tqdm
from scipy.io import wavfile
from utils import *
import audio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy import signal

class BaseAcousticExtractor():
    """
    Base class of acoustic extractor.

    For each acoustic feature extractor, you have to implement the fuction 'extract_features',
    and if you have some special requirements, you may need to reload the __call__ function.

    Noted that the return value (also the orders) is very important for later feature parser.
    """
    def __init__(self, hparams, args):
        super(BaseAcousticExtractor, self).__init__()
        self.hparams = hparams
        self.args = args
        self.base_out_dir = args.out_feature_dir
        os.makedirs(self.base_out_dir, exist_ok=True)
        self.executor = ProcessPoolExecutor(max_workers=args.n_jobs)

    def extract_features(self, label_dict, wav_dir):
        raise NotImplementedError(
            "You need to implement extract_features function based on your feature.")

    def __call__(self, label_dict, wav_dir):
        print('Start to extract {} to {}.'.format(self.hparams.acoustic_type, self.out_dir))
        # Use label dict other than directly wav list, because we may
        # not use all wav data.
        acoustic_metadata = self.extract_features(label_dict, wav_dir)
        return acoustic_metadata


class MelExtractor(BaseAcousticExtractor):
    """
    Mel-spectrogram extractor.

    A spectific mel feature extractor for extracting mel-spectrogram, and it also acts as
    an example for your own extractor.
    """
    def __init__(self, hparams, args):
        BaseAcousticExtractor.__init__(self, hparams, args)
        self.out_dir = os.path.join(self.base_out_dir, 'mels')
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_wav_dir = os.path.join(self.base_out_dir, 'aligned_wavs')
        os.makedirs(self.out_wav_dir, exist_ok=True)
        self.acoustic_description = ["wav_filename", "acoustic_filename", "n_samples", "n_frames"]

    def extract_features(self, label_dict, wav_dir):
        futures = []
        for key in label_dict:
            wav_path = os.path.join(wav_dir, key + '.wav')
            out_wav_path = os.path.join(self.out_wav_dir, key + '.wav')
            futures.append(self.executor.submit(partial(_extract_mel, wav_path, out_wav_path,
                self.out_dir, key, self.hparams, self.args)))
        return [future.result() for future in tqdm(futures) if future.result()]

class MelAudioExtractor(BaseAcousticExtractor):
    """
    Mel-spectrogram extractor.

    A spectific mel feature extractor for extracting mel-spectrogram, and it also acts as
    an example for your own extractor.
    """
    def __init__(self, hparams, args):
        BaseAcousticExtractor.__init__(self, hparams, args)
        self.out_dir = os.path.join(self.base_out_dir, 'mels')
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_wav_dir = os.path.join(self.base_out_dir, 'audio')
        os.makedirs(self.out_wav_dir, exist_ok=True)
        self.acoustic_description = ["wav_filename", "acoustic_filename", "n_samples", "n_frames"]

    def extract_features(self, label_dict, wav_dir):
        futures = []
        for key in label_dict:
            wav_path = os.path.join(wav_dir, key + '.wav')
            out_wav_path = os.path.join(self.out_wav_dir, key + '.wav')
            futures.append(self.executor.submit(partial(_extract_mel_audio, wav_path, self.out_wav_dir,
                self.out_dir, key, self.hparams, self.args)))
        return [future.result() for future in tqdm(futures) if future.result()]

class LinearExtractor(BaseAcousticExtractor):
    """
    Mel-spectrogram extractor.

    A spectific mel feature extractor for extracting mel-spectrogram, and it also acts as
    an example for your own extractor.
    """
    def __init__(self, hparams, args):
        BaseAcousticExtractor.__init__(self, hparams, args)
        self.out_dir = os.path.join(self.base_out_dir, 'mels')
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_wav_dir = os.path.join(self.base_out_dir, 'aligned_wavs')
        os.makedirs(self.out_wav_dir, exist_ok=True)
        self.acoustic_description = ["wav_filename", "acoustic_filename", "n_samples", "n_frames"]

    def extract_features(self, label_dict, wav_dir):
        futures = []
        for key in label_dict:
            wav_path = os.path.join(wav_dir, key + '.wav')
            out_wav_path = os.path.join(self.out_wav_dir, key + '.wav')
            futures.append(self.executor.submit(partial(_extract_linear, wav_path, out_wav_path,
                self.out_dir, key, self.hparams, self.args)))
        return [future.result() for future in tqdm(futures) if future.result()]

class Lf0Extractor(BaseAcousticExtractor):
    """
    Lf0  extractor.

    A spectific mel feature extractor for extracting mel-spectrogram,lf0,uv and energy, and it also acts as
    an example for your own extractor.
    """
    def __init__(self, hparams, args):
        BaseAcousticExtractor.__init__(self, hparams, args)
        print("LF0")
        self.out_dir = os.path.join(self.base_out_dir, 'lf0')
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_lf0_dir = os.path.join(self.base_out_dir, 'lf0')
        os.makedirs(self.out_lf0_dir, exist_ok=True)
        self.acoustic_description = ["wav_filename", "lf0_filename", "n_samples", "n_frames"]

    def extract_features(self, label_dict, wav_dir):
        futures = []
        for key in tqdm(label_dict):
            wav_path = os.path.join(wav_dir, key + '.wav')
            mel_filename = os.path.join(self.out_lf0_dir, key + '.npy')
            if os.path.exists(mel_filename):
                print('skip {}'.format(mel_filename))
                continue
            futures.append(self.executor.submit(partial(_extract_lf0, wav_path, self.out_lf0_dir, key, self.hparams, self.args)))
        return [future.result() for future in tqdm(futures) if future.result()]

class MelLf0UVEnergyExtractor(BaseAcousticExtractor):
    """
    Mel Lf0 uv energy  extractor.

    A spectific mel feature extractor for extracting mel-spectrogram,lf0,uv and energy, and it also acts as
    an example for your own extractor.
    """
    def __init__(self, hparams, args):
        BaseAcousticExtractor.__init__(self, hparams, args)
        print("MEL-LF0-UV-ENERGY")
        self.out_dir = os.path.join(self.base_out_dir, 'mels')
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_wav_dir = os.path.join(self.base_out_dir, 'aligned_wavs')
        os.makedirs(self.out_wav_dir, exist_ok=True)
        self.out_lf0_dir = os.path.join(self.base_out_dir, 'lf0')
        os.makedirs(self.out_lf0_dir, exist_ok=True)
        self.out_uv_dir = os.path.join(self.base_out_dir, 'uv')
        os.makedirs(self.out_uv_dir, exist_ok=True)
        self.out_energy_dir = os.path.join(self.base_out_dir, 'energy')
        os.makedirs(self.out_energy_dir, exist_ok=True)
        self.acoustic_description = ["wav_filename", "acoustic_filename", "n_samples", "n_frames","lf0_filename","uv_filename","energy_filename"]

    def extract_features(self, label_dict, wav_dir):
        futures = []
        for key in label_dict:
            wav_path = os.path.join(wav_dir, key + '.wav')
            out_wav_path = os.path.join(self.out_wav_dir, key + '.wav')
            mel_filename = os.path.join(self.out_uv_dir, key + '.npy')
            if os.path.exists(mel_filename):
                print('skip {}'.format(mel_filename))
                continue
            futures.append(self.executor.submit(partial(_extract_mel_lf0_uv_energy, wav_path, out_wav_path,
                self.out_dir, self.out_lf0_dir, self.out_uv_dir, self.out_energy_dir, key, self.hparams, self.args)))
        return [future.result() for future in tqdm(futures) if future.result()]

def _extract_mel_audio(wav_filename, out_wav_path, out_dir, key, hparams, args):
    wav = audio.load_wav(wav_filename,
                         raw_sr=args.raw_sr,
                         target_sr=hparams.sample_rate,
                         win_size=hparams.win_size,
                         hop_size=hparams.hop_size)
    wav_filename = os.path.join(out_wav_path, key + '.npy')
    np.save(wav_filename, wav, allow_pickle=False)
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    if hparams.trim_silence:
        wav = audio.trim_silence(wav, hparams)
    n_samples = len(wav)

    # Extract mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    n_frames = mel_spectrogram.shape[1]
    if n_frames > hparams.max_acoustic_length:
        print("Ignore wav {} because the frame number {} is too long (Max {} frames in hparams.yaml)."
              .format(wav_filename, n_frames, hparams.max_acoustic_length))
        return None

    # Align features
    desired_frames = int(min(n_samples / hparams.hop_size, n_frames))
    wav = wav[:desired_frames * hparams.hop_size]
    mel_spectrogram = mel_spectrogram[:, :desired_frames]
    n_samples = wav.shape[0]
    n_frames = mel_spectrogram.shape[1]
    assert(n_samples / hparams.hop_size == n_frames)

    # Save intermediate acoustic features
    mel_filename = os.path.join(out_dir, key + '.npy')
    np.save(mel_filename, mel_spectrogram.T, allow_pickle=False)
    # audio.save_wav(wav, out_wav_path, hparams)

    return (wav_filename, mel_filename, n_samples, n_frames)

def _extract_mel(wav_filename, out_wav_path, out_dir, key, hparams, args):
    wav = audio.load_wav(wav_filename,
                         raw_sr=args.raw_sr,
                         target_sr=hparams.sample_rate,
                         win_size=hparams.win_size,
                         hop_size=hparams.hop_size)
    # Process wav samples
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    if hparams.trim_silence:
        wav = audio.trim_silence(wav, hparams)
    n_samples = len(wav)

    # Extract mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    n_frames = mel_spectrogram.shape[1]
    if n_frames > hparams.max_acoustic_length:
        print("Ignore wav {} because the frame number {} is too long (Max {} frames in hparams.yaml)."
              .format(wav_filename, n_frames, hparams.max_acoustic_length))
        return None

    # Align features
    desired_frames = int(min(n_samples / hparams.hop_size, n_frames))
    wav = wav[:desired_frames * hparams.hop_size]
    mel_spectrogram = mel_spectrogram[:, :desired_frames]
    n_samples = wav.shape[0]
    n_frames = mel_spectrogram.shape[1]
    assert(n_samples / hparams.hop_size == n_frames)

    # Save intermediate acoustic features
    mel_filename = os.path.join(out_dir, key + '.npy')
    np.save(mel_filename, mel_spectrogram.T, allow_pickle=False)
    # audio.save_wav(wav, out_wav_path, hparams)

    return (wav_filename, mel_filename, n_samples, n_frames)

def _extract_linear(wav_filename, out_wav_path, out_dir, key, hparams, args):
    wav = audio.load_wav(wav_filename,
                         raw_sr=args.raw_sr,
                         target_sr=hparams.sample_rate,
                         win_size=hparams.win_size,
                         hop_size=hparams.hop_size)
    # Process wav samples
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    if hparams.trim_silence:
        wav = audio.trim_silence(wav, hparams)
    n_samples = len(wav)

    # Extract linear spectrogram
    linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
    n_frames = linear_spectrogram.shape[1]
    if n_frames > hparams.max_acoustic_length:
        print("Ignore wav {} because the frame number {} is too long (Max {} frames in hparams.yaml)."
              .format(wav_filename, n_frames, hparams.max_acoustic_length))
        return None

    # Align features
    desired_frames = int(min(n_samples / hparams.hop_size, n_frames))
    wav = wav[:desired_frames * hparams.hop_size]
    linear_spectrogram = linear_spectrogram[:, :desired_frames]
    n_samples = wav.shape[0]
    n_frames = linear_spectrogram.shape[1]
    assert(n_samples / hparams.hop_size == n_frames)

    # Save intermediate acoustic features
    mel_filename = os.path.join(out_dir, key + '.npy')
    np.save(mel_filename, linear_spectrogram.T, allow_pickle=False)
    # audio.save_wav(wav, out_wav_path, hparams)

    return (wav_filename, mel_filename, n_samples, n_frames)

def _extract_lf0(wav_filename, lf0_out_dir, key, hparams, args, feature_exist=False,extract_lf0=False):
    wav = audio.load_wav(wav_filename,
                         raw_sr=args.raw_sr,
                         target_sr=hparams.sample_rate,
                         win_size=hparams.win_size,
                         hop_size=hparams.hop_size)
    # Process wav samples
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    if hparams.trim_silence:
        wav = audio.trim_silence(wav, hparams)
    n_samples = len(wav)
    lf0_filename = os.path.join(lf0_out_dir, key + '.npy')
    out_wav_path = wav_filename

    #extract lf0
    sound = wav.astype(np.double)
    f0, sp, ap = pw.wav2world(sound, hparams.sample_rate, frame_period=hparams.hop_size/hparams.sample_rate*1000)
    lf0 = np.log(f0 + 1e-8)
    lf0[lf0 < 1e-3] = 0
    lf0 = lf0.astype(np.float32)
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
    # n_frames = lf0_data.shape[1]
    np.save(lf0_filename, lf0_data.astype(np.float32), allow_pickle=False)

    return (wav_filename, None, n_samples, n_samples / hparams.hop_size, lf0_filename, out_wav_path)


def _extract_mel_lf0_uv_energy(wav_filename, save_wavs_path, mel_out_dir, lf0_out_dir,
                              uv_out_dir, energy_out_dir, key, hparams, args, feature_exist=False,extract_lf0=False):
    wav = audio.load_wav(wav_filename,
                         raw_sr=args.raw_sr,
                         target_sr=hparams.sample_rate,
                         win_size=hparams.win_size,
                         hop_size=hparams.hop_size)
    # Process wav samples
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    if hparams.trim_silence:
        wav = audio.trim_silence(wav, hparams)
    n_samples = len(wav)
    mel_filename = os.path.join(mel_out_dir, key + '.npy')
    energy_filename = os.path.join(energy_out_dir, key + '.npy')
    lf0_filename = os.path.join(lf0_out_dir, key + '.npy')
    uv_filename = os.path.join(uv_out_dir, key + '.npy')
    out_wav_path = wav_filename
    if feature_exist:
        # extract lf0,uv
        if extract_lf0:
            sound = wav.astype(np.double)
            f0, sp, ap = pw.wav2world(sound, hparams.sample_rate,
                                      frame_period=hparams.hop_size / hparams.sample_rate * 1000)
            lf0 = np.log(f0 + 1e-8)
            lf0[lf0 < 1e-3] = 0
            lf0 = lf0.astype(np.float32)

            def interpolate_f0(data):
                data = np.reshape(data, (data.size, 1))
                vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
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

            lf0_data, uv = interpolate_f0(lf0.copy())

            if hparams.lf0_norm == 'minmax':
                lf0_data = (lf0_data - min(lf0_data)) / (max(lf0_data) - min(lf0_data) + 1e-6)
            elif hparams.lf0_norm == 'z-score':
                mean = np.mean(lf0_data[uv > 0.0])
                std = np.std(lf0_data[uv > 0.0])
                lf0_data = (lf0_data - mean) / std * uv
                lf0_data[uv <= 0.0] = -4.0
            elif hparams.lf0_norm == 'raw':
                lf0_data = lf0
            else:
                raise Exception('{} does\'t support now!'.format(hparams.lf0_norm))
            np.save(lf0_filename, lf0_data.astype(np.float32), allow_pickle=False)
        return (
        wav_filename, mel_filename, n_samples, n_samples / hparams.hop_size, lf0_filename, uv_filename, energy_filename, out_wav_path)

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
    np.save(mel_filename, mel_spectrogram.T, allow_pickle=False)
    if hparams.save_align_wavs:
        audio.save_wav(wav, save_wavs_path, hparams)
    else:
        out_wav_path = wav_filename

    #extract lf0,uv
    sound = wav.astype(np.double)
    f0, sp, ap = pw.wav2world(sound, hparams.sample_rate, frame_period=hparams.hop_size/hparams.sample_rate*1000)
    lf0 = np.log(f0 + 1e-8)
    lf0[lf0 < 1e-3] = 0
    lf0 = lf0.astype(np.float32)
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
    np.save(lf0_filename, lf0_data.astype(np.float32), allow_pickle=False)
    np.save(uv_filename, uv.astype(np.float32), allow_pickle=False)

    #extract energy minmax
    def extract_energy(wav,window_size,shift_size):
        energys = []
        for i in range(0, len(wav) - window_size, shift_size):
            power = np.sum(np.absolute(wav[i: i + window_size])) / window_size
            power = np.clip(power, 1e-9, 1e9)
            energys.append([np.log10(power)])
        return np.array(energys)
    energy = extract_energy(wav, hparams.win_size, hparams.hop_size)
    # energy = librosa.feature.rms(y=wav,frame_length=hparams.win_size,hop_length=hparams.hop_size).reshape(-1,1)
    # print(energy)
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
        raise Exception('{} does\'t support now!'.format(hparams.lf0_norm))

    # print(energy)
    np.save(energy_filename, energy.astype(np.float32), allow_pickle=False)

    return (wav_filename, mel_filename, n_samples, n_frames, lf0_filename, uv_filename, energy_filename, out_wav_path)

class FeatureExtractor():
    def __init__(self, hparams, args):
        super(FeatureExtractor, self).__init__()
        self.args = args
        self.hparams = hparams
        self.acoustic_type = hparams.acoustic_type
        self.wav_dir = args.wav_dir
        assert (self.acoustic_type in hparams.supported_acoustic_type)
        self.supported_extractors = {"Mel": MelExtractor,
                                     "Linear": LinearExtractor,
                                     "Lf0":Lf0Extractor,
                                     "Mel-Audio":MelAudioExtractor,
                                     "Mel-Lf0-UV-Energy":MelLf0UVEnergyExtractor}
        self.extractor = self.supported_extractors[self.acoustic_type](hparams, args)

    def __call__(self, label_dict):
        acoustic_metadata = self.extractor(label_dict, self.wav_dir)
        total_frames = sum([int(m[3]) for m in acoustic_metadata])
        total_samples = sum([int(m[2]) for m in acoustic_metadata])
        sr = self.hparams.sample_rate
        hours = total_samples / sr / 3600.0

        print("Successfully extract {} utterances, about {:.2f} hours."
              .format(len(acoustic_metadata), hours))
        print('Max acoustic frames length: {}'.format(
            max(int(m[3]) for m in acoustic_metadata)))

        random.shuffle(acoustic_metadata)
        acoustic_dict = {}
        for single_feature in acoustic_metadata:
            feature_index = os.path.splitext(os.path.basename(single_feature[0]))[0]
            acoustic_dict[feature_index] = single_feature

        return acoustic_dict
