import numpy as np
import sox
import librosa

import sys
from tqdm import tqdm
import torch
torch.multiprocessing.set_start_method('spawn', force=True)
import numpy as np
import os
import torchaudio.compliance.kaldi as kaldi
from functools import partial
import argparse
torch.set_num_threads(1)

from aslp_utils import LanceWriter, LanceReader, AudioData, FloatNPYData
from multiprocessing import Pool
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

parser = argparse.ArgumentParser()
parser.add_argument("data_root", type=str, help="wav dir")
parser.add_argument("out_data_root", type=str, help="out bn feature dir")
parser.add_argument("--filelist", type=str, help="filelst", default=None)
parser.add_argument("--use_lance", type=bool, default=False)
parser.add_argument("--tempo_factor", type=float, default=1.5)

# data_root = sys.argv[1]
# out_data_root = sys.argv[2]
args = parser.parse_args()
print(args)

data_root = args.data_root
out_data_root = args.out_data_root
filelist = args.filelist
use_lance = args.use_lance
tempo_factor = args.tempo_factor

# def inference():
#     # 假设你已经有了加载的音频数据
#     audio_path = '/home/work_nfs16/zhguo/code/dualvc/testdata/wavs/SSB00730313.wav'
#     audio, sr = librosa.load(audio_path, sr=16000)
#     audio = (audio * np.iinfo(np.int16).max).astype(np.int16)

#     # 创建一个Transformer对象
#     tfm = sox.Transformer()

#     # 设置变速不变调，speed_factor为速度因子，例如1.5表示加速1.5倍
#     tfm.tempo(factor=1.5, audio_type='s')

#     # 使用build_array方法直接处理numpy数组
#     processed_audio = tfm.build_array(input_array=audio, sample_rate_in=sr)

#     # 如果需要将处理后的音频数据转换回浮点数格式
#     # processed_audio = processed_audio.astype(np.float32) / np.iinfo(np.int16).max

#     output_path = '/home/work_nfs16/zhguo/code/dualvc/test_tempo.wav'  # 替换为你想保存的路径
#     # sf.write(output_path, processed_audio, sr)
    
tfm = sox.Transformer()
tfm.tempo(factor=tempo_factor, audio_type='s')

def tempo(wav):
    if type(wav) == AudioData:
        audio = wav.audio
        sr = wav.sample_rate
        utt = wav.data_id
        duration = wav.duration
        
        processed_audio = tfm.build_array(input_array=audio, sample_rate_in=sr)
        new_utt = f"{utt}_temp{tempo_factor}"
        
        data = AudioData(data_id=new_utt, audio=processed_audio, sample_rate=sr, duration=duration)
        return data
    else:
        assert NotImplementedError


def tempo_lance(row, wav_reader: LanceReader):
    
    try:
        data = tempo(wav_reader.get_datas_by_rows([row])[0])
    except Exception as e:
        print(f"error:{row}")
        print(e)
        return None
    return data
    

if __name__ == "__main__":


    if not use_lance:
        # os.makedirs(out_data_root, exist_ok=True)

        # if filelist is not None:
        #     # filelist = sys.argv[3]
        #     #print("file!!", filelist)
        #     filelist = open(filelist).read().splitlines()
        #     generator = filelist
        #     #print(generator[:10])
        # else:
        #     generator = os.listdir(data_root)

        # for wav_filename in tqdm(generator):
        #     wav_path = os.path.join(data_root, wav_filename)
        #     try:
        #         bn = extract_bn(wav_path)
        #     except:
        #         print("!!!!!!!", wav_path)
        #     np.save(os.path.join(out_data_root, wav_filename.split('.')[0]), bn)
        
        assert NotImplementedError

    else:
        
        WRITE_INTERVAL = 10000
        
        wav_reader = LanceReader(data_root, target_cls=AudioData)
        writer = LanceWriter(out_data_root, target_cls=AudioData)
        
        print("readling data ids")
        wav_ids = wav_reader.get_ids()
        rows = list(range(len(wav_ids)))
        print("data ids loaded")
        
        wav_data = []

        # import pdb;pdb.set_trace()
        tempo_function = partial(tempo_lance, wav_reader=wav_reader)

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
        