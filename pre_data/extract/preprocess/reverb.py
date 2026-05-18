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
import sox
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

tfm = sox.Transformer()
tfm.reverb()


# 随机生成混响参数的函数
def random_reverb_params():
    reverberance = random.uniform(20, 80)  # 混响强度，20到80之间的随机值
    high_freq_damping = random.uniform(20, 80)  # 高频衰减
    room_scale = random.uniform(75, 100)  # 房间大小
    pre_delay = random.uniform(10, 30)  # 预延迟
    return reverberance, high_freq_damping, room_scale, pre_delay


def reverb_lance(row, reader: LanceReader):
    # import pdb;pdb.set_trace()
    wav = reader.get_datas_by_rows([row])[0]

    audio = wav.audio
    sr = wav.sample_rate
    utt = wav.data_id
    duration = wav.duration

    # 随机生成混响参数
    reverberance, high_freq_damping, room_scale, pre_delay = random_reverb_params()

    # 创建 Transformer 对象
    tfm = sox.Transformer()

    # 添加混响效果，使用随机参数
    tfm.reverb(reverberance=reverberance, 
                high_freq_damping=high_freq_damping, 
                room_scale=room_scale, 
                pre_delay=pre_delay)
    
    processed_audio = tfm.build_array(input_array=audio, sample_rate_in=sr)
    
    data = AudioData(data_id=utt+'_reverb', audio=processed_audio, sample_rate=sr, duration=duration)
    return data


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
        tempo_function = partial(reverb_lance, reader=wav_reader)

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
        