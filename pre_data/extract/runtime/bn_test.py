import torch
import numpy as np
import os
from vc.utils import audio

asr = torch.jit.load('runtime/models/fastu2++.pt')

device='cpu'
#feats = torch.from_numpy(np.load('feats1.npy')).to(device)
#feats = feats.to(device)

#np.save('feats.npy', feats.cpu().numpy(), allow_pickle=False)

wav = audio.load_wav("1.wav", 16000)
feats = audio.fbanks(wav, frame_shift=10)

xs = feats
decoding_chunk_size = 4
num_decoding_left_chunks = 2
subsampling = 4
context = 7  # Add current frame
stride = subsampling * decoding_chunk_size
decoding_window = (decoding_chunk_size - 1) * subsampling + context
num_frames = xs.size(1)
att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
outputs = []
offset = 0
required_cache_size = decoding_chunk_size * num_decoding_left_chunks

# Feed forward overlap input step by step
for cur in range(0, num_frames - context + 1, stride):
    end = min(cur + decoding_window, num_frames)
    chunk_xs = xs[:, cur:end, :]
    #(y, att_cache, cnn_cache) = encoder_ts.forward_chunk(
    (y, att_cache, cnn_cache) = asr.forward_encoder_chunk(
        chunk_xs, offset, required_cache_size, att_cache, cnn_cache)
    outputs.append(y)

    offset += y.size(1)
    #break
ys = torch.cat(outputs, 1)
encoder_out = ys

encoder_out = encoder_out.transpose(1, 2)
encoder_out = torch.nn.functional.interpolate(encoder_out, size=int(encoder_out.shape[2] * 16 / 5), mode='linear', align_corners=True)
encoder_output = encoder_out.transpose(1, 2)

encoder_output = encoder_output.data
encoder_output = encoder_output.cpu().numpy()[0]
npy_path = os.path.join('testdata/test/bn', '{}.npy'.format('1'))
np.save(npy_path, encoder_output, allow_pickle=False)