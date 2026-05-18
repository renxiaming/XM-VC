import sys
import numpy as np
from tqdm import tqdm
import torch
torch.multiprocessing.set_start_method('spawn', force=True)
import numpy as np
import os
import librosa
import torchaudio.compliance.kaldi as kaldi
from functools import partial
import argparse
torch.set_num_threads(1)

from aslp_utils import LanceWriter, LanceReader, AudioData, FloatNPYData
from multiprocessing import Pool
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


def extract_fbanks(
    wav, sample_rate=16000, mel_bins=80, frame_length=25, frame_shift=12.5
):
    wav = wav * (1 << 15)
    wav = torch.from_numpy(wav).unsqueeze(0)
    fbanks = kaldi.fbank(
        wav,
        frame_length=frame_length,
        frame_shift=frame_shift,
        snip_edges=True,
        num_mel_bins=mel_bins,
        energy_floor=0.0,
        dither=0.0,
        sample_frequency=sample_rate,
    )
    # fbanks = mat.detach().numpy()
    fbanks = fbanks.unsqueeze(0)
    return fbanks


asr = torch.jit.load('runtime/models/fastu2++.pt').to(device)


def extract_bn(wav_path, use_lance=False):
    
    if not use_lance:
        in_wav, _ = librosa.load(wav_path, sr=16000)
    else:
        # wav_path: AudioData
        in_wav = wav_path.audio
        in_wav = in_wav / np.iinfo(in_wav.dtype).max

        if wav_path.sample_rate != 16000:
            if wav_path.sample_rate < 16000:
                print(f"warning: Resample from {wav_path.sample_rate} to 16000")
            in_wav = librosa.core.resample(
                in_wav, 
                orig_sr=wav_path.sample_rate,
                target_sr=16000, 
                res_type='kaiser_best')

    fbanks = extract_fbanks(in_wav, frame_shift=10).float().to(device)
    #print(in_wav.shape, fbanks.shape)
    offset = 0
    decoding_chunk_size = 5
    num_decoding_left_chunks = 2
    subsampling = 4
    context = 7  # Add current frame
    stride = subsampling * decoding_chunk_size
    required_cache_size = decoding_chunk_size * num_decoding_left_chunks
    decoding_window = (decoding_chunk_size - 1) * subsampling + context
    att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device='cpu')
    cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device='cpu')
    #for i in tqdm(range(0, fbanks.shape[1], 20)):
    # bns = []
    # for i in range(0, fbanks.shape[1], 20):
    #     fbank = fbanks[:, i:i+23, :]
    #     if fbank.shape[1] < 10:
    #         break
    #     #print("input", fbank.shape)
    #     (encoder_output, att_cache, cnn_cache) = asr.forward_encoder_chunk(
    #         fbank, offset, required_cache_size, att_cache, cnn_cache)

    #     encoder_output_upsample = encoder_output.transpose(1, 2)
    #     encoder_output_upsample = torch.nn.functional.interpolate(encoder_output_upsample, size=int(encoder_output_upsample.shape[2] * 16 / 5), mode='linear', align_corners=True)
    #     encoder_output_upsample = encoder_output_upsample.transpose(1, 2)
    #     #print(fbank.shape, encoder_output.shape, encoder_output_upsample.shape)

    #     bns.append(encoder_output_upsample)

    #     offset += encoder_output.size(1)
    
    # bn = torch.cat(bns, dim=1)
    # bn = bn.squeeze()
    # print("---- streaming fbank -----", bn.shape)
    # bn = bn.detach().cpu().numpy()
    # (encoder_output, att_cache, cnn_cache) = asr.forward_encoder_chunk(
    #         fbanks, offset, required_cache_size, att_cache, cnn_cache)
    print("---- fbnks ----", fbanks.shape)
    encoder_output = asr._forward_encoder(fbanks, fbanks.shape[1])
    print("---- encoder_output ----", encoder_output.shape)
    (encoder_output, att_cache, cnn_cache) = asr.forward_encoder_chunk(
            fbanks, offset, required_cache_size, att_cache, cnn_cache)
    print("---- encoder_output_old ----", encoder_output.shape)
    encoder_output_upsample = encoder_output.transpose(1, 2)
    encoder_output_upsample = torch.nn.functional.interpolate(encoder_output_upsample, size=int(encoder_output_upsample.shape[2] * 16 / 5), mode='linear', align_corners=True)
    encoder_output_upsample = encoder_output_upsample.transpose(1, 2)
    bn = encoder_output_upsample.squeeze()
    print("---- unstreaming fbank -----", bn.shape)
    bn = bn.detach().cpu().numpy()
    
    #print(bn.shape)
    if use_lance:
        bn = FloatNPYData(wav_path.data_id, data=bn)
    return bn

# def extract_bn_lance(wav_data):
#     return extract_bn(wav_data, use_lance=True)

def extract_bn_lance(rowid, reader:LanceReader):
    try:
        bn = extract_bn(reader.get_datas_by_rowids([rowid])[0], use_lance=True)
    except Exception as e:
        print(f"error:{rowid}")
        print(e)
        print("error bn")
        return None
    return bn

# python preprocess/extract_bn.py /home/work_nfs16/zhguo/data/hq_cn_lance_data_tempo1.5 /home/work_nfs16/zhguo/data/hq_cn_bn_lance_data_tempo1.5 --use_lance true
# python preprocess/extract_bn.py /home/work_nfs16/zhguo/data/hq_cn_lance_data_tempo0.8 /home/work_nfs16/zhguo/data/hq_cn_bn_lance_data_tempo0.8 --use_lance true
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str, help="wav dir")
    parser.add_argument("out_data_root", type=str, help="out bn feature dir")
    parser.add_argument("--filelist", type=str, help="filelst", default=None)
    parser.add_argument("--use_lance", type=bool, default=False)

    # data_root = sys.argv[1]
    # out_data_root = sys.argv[2]
    args = parser.parse_args()

    data_root = args.data_root
    out_data_root = args.out_data_root
    filelist = args.filelist
    use_lance = args.use_lance

    if not use_lance:
        os.makedirs(out_data_root, exist_ok=True)

        if filelist is not None:
            # filelist = sys.argv[3]
            #print("file!!", filelist)
            filelist = open(filelist).read().splitlines()
            generator = filelist
            #print(generator[:10])
        else:
            generator = os.listdir(data_root)

        for wav_filename in tqdm(generator):
            wav_path = os.path.join(data_root, wav_filename)
            try:
                bn = extract_bn(wav_path)
            except:
                print("!!!!!!!", wav_path)
            np.save(os.path.join(out_data_root, wav_filename.split('.')[0]), bn)

    else:
        
        WRITE_INTERVAL = 10000
        wav_reader = LanceReader(data_root, target_cls=AudioData)
        writer = LanceWriter(out_data_root, target_cls=FloatNPYData)
        
        wav_ids = wav_reader.get_ids()
        
        # rows = list(range(len(wav_ids)))

        processed_ids = writer.get_ids()
        processed_ids = set([id.data_id for id in processed_ids])
        wav_ids = [wav_id for wav_id in wav_ids if wav_id.data_id not in processed_ids]
        rowids = [wav_id._rowid for wav_id in wav_ids]

        bn_data = []

        # import pdb;pdb.set_trace()
        bn_function = partial(extract_bn_lance, reader=wav_reader)

        with Pool(20) as pool:
            for i in tqdm(
                pool.imap_unordered(bn_function, rowids),
                total=len(rowids)
            ):
                if i == None:
                    continue
                bn_data.append(i)
                if len(bn_data) > WRITE_INTERVAL:
                    writer.write_parallel(bn_data)
                    bn_data = []
        writer.write_parallel(bn_data)
        
        # for row in tqdm(rows):
            
        #     wav_data = wav_reader.get_datas_by_rows([row])[0]
        #     try:
        #         bn = extract_bn(wav_data, use_lance=True)
        #     except:
        #         print("!!!!!!!", wav_data.data_id)
        #     bn = FloatNPYData(wav_data.data_id, data=bn)
        #     bn_data.append(bn)
        #     if len(bn_data) > WRITE_INTERVAL:
        #         writer.write_parallel(bn_data)
        #         bn_data = []

