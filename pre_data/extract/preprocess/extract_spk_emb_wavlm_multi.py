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
import torch.multiprocessing as mp
# from multiprocessing import Pool
# import multiprocessing
mp.set_start_method('spawn', force=True)

# MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]
MODEL_LIST = ["wavlm_large"]


def init_model(model_name, checkpoint=None):
    if model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    else:
        assert True

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model


       
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', required=True)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-l', "--use_lance", type=int, default=1)
parser.add_argument('-j', "--num_thread_per_gpu", type=int, default=10)
args = parser.parse_args()

wav_dir = args.input_dir
output = args.output
use_lance = args.use_lance
num_thread = args.num_thread_per_gpu
print(args)

def inference(rank, queue_input:mp.Queue, queue_output:mp.Queue, wav_reader:LanceReader, sample_rate=16000):
    device = f"cuda:{rank}"

    model = init_model('wavlm_large', 'preprocess/ckpts/wavlm_large_finetune.pth')
    model.eval()
    model.to(device)
    print(f"model loaded to {device}")

    while True:
        wav_id:AudioData = queue_input.get()
        if wav_id is None:
            queue_output.put(None)
            break
        data:AudioData = wav_reader.get_datas_by_rowids([wav_id._rowid])[0]
        wav, sr = data.audio, data.sample_rate

        wav = torch.from_numpy(wav).unsqueeze(0).float().to(device)

        with torch.no_grad():
            if sr != sample_rate:
                resample = Resample(orig_freq=sr, new_freq=sample_rate).to(device)
                wav = resample(wav)

            emb = model(wav)
            emb = emb.squeeze(0).detach().cpu().numpy()
            # print(emb.shape)
        queue_output.put(FloatNPYData(wav_id.data_id, data=emb))
    


if __name__ == "__main__":

    gpu_num = torch.cuda.device_count()
    num_threads = num_thread * gpu_num


    processes = []

    queue_input:mp.Queue = mp.Queue()
    queue_output:mp.Queue = mp.Queue()

    if use_lance == 1:
        
        WRITE_INTERVAL = 10000
        
        wav_reader = LanceReader(wav_dir, target_cls=AudioData)
        writer = LanceWriter(output, target_cls=FloatNPYData)

        for thread_num in range(num_threads):
            
            rank = thread_num % gpu_num
            print(num_threads, rank)
            p = mp.Process(target=inference, args=(rank, queue_input, queue_output, wav_reader), daemon=True)
            p.start()
            processes.append(p)
            
        wav_ids = wav_reader.get_ids()
        processed_ids = writer.get_ids()
        processed_ids = set([id.data_id for id in processed_ids])
        wav_ids = [wav_id for wav_id in wav_ids if wav_id.data_id not in processed_ids]

        for wav_id in tqdm(wav_ids, desc='add data to queue'):
            queue_input.put(wav_id)

        for _ in range(num_threads):
            queue_input.put(None)

        spk_data = []
        for _ in tqdm(range(len(wav_ids) + num_threads), desc='get inference output'):
            data = queue_output.get()
            if data is None:
                continue

            spk_data.append(data)

            if len(spk_data) > WRITE_INTERVAL:
                writer.write_parallel(spk_data)
                spk_data = []
        writer.write_parallel(spk_data)

    
    else:
        assert True
        

    
    # if use_lance == 1:
        
        
    #     # generate_embs(model, row, reader:LanceReader, device='cpu', sample_rate=16000)
    #     gen_function = partial(generate_embs_lance, reader = wav_reader, device=device, sample_rate=16000)

    #     with Pool(num_thread) as pool:
    #         for i in tqdm(
    #             pool.imap_unordered(gen_function, rows),
    #             total=len(rows)
    #         ):
    #             if i == None:
    #                 continue
    #             spk_data.append(i)
    #             if len(spk_data) > WRITE_INTERVAL:
    #                 writer.write_parallel(spk_data)
    #                 spk_data = []
    #     writer.write_parallel(spk_data)

        
    # else:
    #     wavs = glob.glob(os.path.join(wav_dir, '*.wav'))
    #     input_args = []
    #     for wav in wavs:
    #         utt = wav.split('/')[-1][:-4]
    #         out_path = os.path.join(output, utt+'.npy')
    #         input_args.append((wav, out_path))

    #     gen_function = partial(generate_embs_file, device=device, sample_rate=16000)

    #     with Pool(num_thread) as pool:
    #         for i in tqdm(
    #             pool.imap_unordered(gen_function, input_args),
    #             total=len(input_args)
    #         ):
    #             pass

