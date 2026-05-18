import soundfile as sf
import torch
import os
from collections import defaultdict 
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL
import glob
import numpy as np
from functools import partial

from aslp_utils import LanceWriter, LanceReader, AudioData, FloatNPYData
from multiprocessing import Pool
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]
MODEL_LIST = ["wavlm_large"]


def init_model(model_name, checkpoint=None):
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
    elif model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=config_path)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model


       
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--output')
parser.add_argument('--device', default="cuda:0")
parser.add_argument("--use_lance", type=int, default=1)
parser.add_argument("--num_thread", type=int, default=10)
args = parser.parse_args()

wav_dir = args.input_dir
output = args.output
device = args.device
use_lance = args.use_lance
num_thread = args.num_thread
print(args)

model = init_model('wavlm_large', 'preprocess/ckpts/wavlm_large_finetune.pth')
model.eval()
model.to(device) 
print(f"model loaded to {device}")

def get_emb(wav, device='cpu', sample_rate=16000):
    if type(wav) == str:
        wav, sr = sf.read(wav)

        wav = torch.from_numpy(wav).unsqueeze(0).float().to(device)
    elif type(wav) == AudioData:
        sr = wav.sample_rate
        wav = wav.audio
        wav = wav / np.iinfo(wav.dtype).max
        wav = torch.from_numpy(wav).unsqueeze(0).float().to(device)

    if sr != sample_rate:
        resample = Resample(orig_freq=sr, new_freq=sample_rate).to(device)
        wav = resample(wav)

    with torch.no_grad():
        emb = model(wav)
        emb = emb.squeeze(0).detach().cpu().numpy()
        # print(emb.shape)

    return emb


# 包装为Lance的处理流程
def generate_embs_lance(row, reader:LanceReader, device='cpu', sample_rate=16000):
    try:
        data = reader.get_datas_by_rows([row])[0]
        emb = get_emb(data, device, sample_rate)
        # print(emb)
        emb = FloatNPYData(data.data_id, data=emb)
        # print(emb)
    except Exception as e:
        print(row, e)
        return None
    return emb

# 普通文件的处理流程
def generate_embs_file(args:tuple, device='cpu', sample_rate=16000):
    file, out_path = args
    emb = get_emb(file, device, sample_rate)
    np.save(out_path, emb)
    return 0
    


if __name__ == "__main__":
    
    if use_lance == 1:
        
        WRITE_INTERVAL = 10000
        
        wav_reader = LanceReader(wav_dir, target_cls=AudioData)
        writer = LanceWriter(output, target_cls=FloatNPYData)
        
        wav_ids = wav_reader.get_ids()
        rows = list(range(len(wav_ids)))
        
        spk_data = []
        # generate_embs(model, row, reader:LanceReader, device='cpu', sample_rate=16000)
        gen_function = partial(generate_embs_lance, reader = wav_reader, device=device, sample_rate=16000)

        with Pool(num_thread) as pool:
            for i in tqdm(
                pool.imap_unordered(gen_function, rows),
                total=len(rows)
            ):
                if i == None:
                    continue
                spk_data.append(i)
                if len(spk_data) > WRITE_INTERVAL:
                    writer.write_parallel(spk_data)
                    spk_data = []
        writer.write_parallel(spk_data)

        
    else:
        wavs = glob.glob(os.path.join(wav_dir, '*.wav'))
        input_args = []
        for wav in wavs:
            utt = wav.split('/')[-1][:-4]
            out_path = os.path.join(output, utt+'.npy')
            input_args.append((wav, out_path))

        gen_function = partial(generate_embs_file, device=device, sample_rate=16000)

        with Pool(num_thread) as pool:
            for i in tqdm(
                pool.imap_unordered(gen_function, input_args),
                total=len(input_args)
            ):
                pass
