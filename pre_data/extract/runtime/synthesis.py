import torch
import numpy as np
import os
from vc.utils import audio

asr = torch.jit.load('runtime/models/fastu2++.pt')
vc = torch.jit.load('cbhgar.pt')
vocoder = torch.jit.load('runtime/models/melgan.pt')

wav = audio.load_wav("1.wav", 16000)
feats = audio.fbanks(wav, frame_shift=10)

#print(wav.shape, wav.min(), wav.max())
print(wav.shape, wav.min(), wav.max(), feats.shape)

feats_list = []

frame_shift = int(16000 / 1000 * 10)
frame_len = int(16000 / 1000 * 25)
wav_cache_len = 24 * 16
wav_cache = np.ones(wav_cache_len) * -0.5
fbank_chunksize = 23 * 160

wav_all = np.concatenate((wav_cache, wav))
feats_all = audio.fbanks(wav_all, frame_shift=10)
print(wav_all.shape, feats_all.shape)

for i in range(0, len(wav), fbank_chunksize):
    end_pos = min(i + fbank_chunksize, len(wav))
    wav_chunk = wav[i:end_pos]
    wav_chunk = np.concatenate((wav_cache, wav_chunk))
    wav_cache = wav_chunk[-wav_cache_len:]
    feats_chunk = audio.fbanks(wav_chunk, frame_shift=10)
    feats_list.append(feats_chunk)

feats_chunk_all = torch.cat(feats_list, dim=1)
print(feats_chunk_all.shape)
torch.testing.assert_close(feats_all, feats_chunk_all)


xs = feats
decoding_chunk_size = 5
num_decoding_left_chunks = 2
subsampling = 4
context = 7  # Add current frame
stride = subsampling * decoding_chunk_size
decoding_window = (decoding_chunk_size - 1) * subsampling + context
num_frames = xs.size(1)
att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
outputs = []
mels = []
wavs = []
offset = 0
required_cache_size = decoding_chunk_size * num_decoding_left_chunks
vc_cache = None
vc_offset = 0

vocoder_cache = None
vocoder_overlap = 2

# Feed forward overlap input step by step
for cur in range(0, num_frames - context + 1, stride):
    end = min(cur + decoding_window, num_frames)
    chunk_xs = xs[:, cur:end, :]
    #(y, att_cache, cnn_cache) = encoder_ts.forward_chunk(
    (encoder_output, att_cache, cnn_cache) = asr.forward_encoder_chunk(
        chunk_xs, offset, required_cache_size, att_cache, cnn_cache)
    outputs.append(encoder_output)
    print(chunk_xs.shape, encoder_output.shape)

    encoder_output_upsample = encoder_output.transpose(1, 2)
    encoder_output_upsample = torch.nn.functional.interpolate(encoder_output_upsample, size=int(encoder_output_upsample.shape[2] * 16 / 5), mode='linear', align_corners=True)
    encoder_output_upsample = encoder_output_upsample.transpose(1, 2)

    offset += encoder_output.size(1)


    spks = torch.LongTensor([[13]])
    generate_output_chunk, vc_cache, ar_input_chunk, ar_prenet_chunk, ar_gru_chunk, ar_slice_gt_input_chunk, ar_hidden_state_chunk = vc.inference(
        encoder_output_upsample,
        spks,
        vc_offset,
        cache=vc_cache,
    )
    vc_offset += 16
    mels.append(generate_output_chunk)

    mel = (generate_output_chunk.transpose(1, 2) + 4.) / 8.

    if vocoder_cache is not None:
        mel = torch.cat([vocoder_cache, mel], dim=-1)
    vocoder_cache = mel[:, :, -vocoder_overlap:]
    wav = vocoder(mel).squeeze()
    wavs.append(wav)
    #print(encoder_output.shape, encoder_output_upsample.shape, generate_output_chunk.shape, wav.shape, vocoder_cache.shape)
    #break
ys = torch.cat(outputs, 1)
mel = torch.cat(mels, 1)
upsample_factor = 200
vocoder_wav_overlap = vocoder_overlap * upsample_factor
last_wav = None
down_linspace = torch.linspace(1, 0, steps=vocoder_wav_overlap, out=None)
up_linspace = torch.linspace(0, 1, steps=vocoder_wav_overlap, out=None)
new_wavs = []
for i, wav in enumerate(wavs):
    if last_wav is not None:
        front_wav = wav[:vocoder_wav_overlap]
        smooth_front_wav = last_wav * down_linspace + front_wav * up_linspace
        #smooth_front_wav = front_wav
        new_wav = torch.cat([smooth_front_wav, wav[vocoder_wav_overlap:-vocoder_wav_overlap]], dim=0)
    else:
        new_wav = wav[:-vocoder_wav_overlap]
    last_wav = wav[-vocoder_wav_overlap:]
    #print(wav.shape, new_wav.shape, last_wav.shape, down_linspace.shape, up_linspace.shape)
    new_wavs.append(new_wav)

#wav = torch.cat(wavs, -1).squeeze()
wav = torch.cat(new_wavs, dim=-1)
encoder_out = ys

encoder_out = encoder_out.transpose(1, 2)
encoder_out = torch.nn.functional.interpolate(encoder_out, size=int(encoder_out.shape[2] * 16 / 5), mode='linear', align_corners=True)
encoder_output = encoder_out.transpose(1, 2)

encoder_output = encoder_output.data
encoder_output = encoder_output.cpu().numpy()[0]
npy_path = os.path.join('testdata/test/bn', '{}.npy'.format('1'))
np.save(npy_path, encoder_output, allow_pickle=False)

generated_acoustic = np.reshape(
    mel.detach().cpu().numpy()[0],
    [-1, 80],
)
acoustic_output_path = os.path.join(
   '.', "{}.npy".format('1')
)
np.save(acoustic_output_path, generated_acoustic, allow_pickle=False)

from scipy.io.wavfile import write
write("1_out.wav", 16000, wav.detach().cpu().numpy())

wav_all = vocoder((mel.transpose(1, 2) + 4.) / 8.).squeeze().detach().cpu().numpy()
write("1_out_all.wav", 16000, wav_all)