import sys
import numpy as np
from tqdm import tqdm
import torch
import numpy as np
import os
import librosa
import torchaudio.compliance.kaldi as kaldi
torch.set_num_threads(1)

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

def extract_bn(wav_path):
    in_wav, _ = librosa.load(wav_path, sr=16000)
    fbanks = extract_fbanks(in_wav, frame_shift=10).float().cpu().numpy()
    #print(in_wav.shape, fbanks.shape)
    return fbanks
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
    bns = []
    for i in range(0, fbanks.shape[1], 20):
        fbank = fbanks[:, i:i+23, :]
        if fbank.shape[1] < 10:
            break
        #print("input", fbank.shape)
        (encoder_output, att_cache, cnn_cache) = asr.forward_encoder_chunk(
            fbank, offset, required_cache_size, att_cache, cnn_cache)

        bns.append(encoder_output)

        offset += encoder_output.size(1)

    bn = torch.cat(bns, dim=1)
    bn = bn.squeeze()
    bn = bn.detach().cpu().numpy()
    #print(bn.shape)
    return bn

data_root = sys.argv[1]
out_data_root = sys.argv[2]

os.makedirs(out_data_root, exist_ok=True)

if len(sys.argv) >= 4:
    filelist = sys.argv[3]
    filelist = open(filelist).read().splitlines()
    generator = filelist
else:
    generator = os.listdir(data_root)

for wav_filename in tqdm(generator):
    wav_path = os.path.join(data_root, wav_filename)
    bn = extract_bn(wav_path)
    np.save(os.path.join(out_data_root, wav_filename.split('.')[0]), bn)
