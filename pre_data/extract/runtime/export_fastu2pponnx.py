import torch 
import librosa
import torchaudio.compliance.kaldi as kaldi
import torch

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

class Asr(torch.nn.Module):
    def __init__(self):
        super(Asr, self).__init__()
        self.asr = torch.jit.load('runtime/models/fastu2++.pt')
    def forward(self, a: torch.Tensor, b: int, c:int, d:torch.Tensor, e:torch.Tensor):
        return self.asr.forward_encoder_chunk(a, b, c, d, e)

#asr = torch.jit.load('runtime/models/fastu2++.pt')
asr = torch.jit.script(Asr())
wav, _ = librosa.load('testdata/eval/wavs/1.wav', sr=16000)
fbanks = extract_fbanks(wav, frame_shift=10).float()
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
#asr.forward = asr.forward_encoder_chunk
for i in range(0, fbanks.shape[1], 20):
    fbank = fbanks[:, i:i+23, :]
    if fbank.shape[1] < 10:
        break
    #print("input", fbank.shape)
    (encoder_output, att_cache, cnn_cache) = asr.forward(
        fbank, offset, required_cache_size, att_cache, cnn_cache)
    print(att_cache.shape, cnn_cache.shape, offset)

    offset += encoder_output.size(1)
dummy_inputs=(fbanks[:, 0:23, :], 10, 10, torch.zeros(7, 4, 10, 128), torch.zeros(7, 1, 256, 8))
input_names=["fbank", "offset", "required_cache_size", "att_cache", "cnn_cache"]
output_names=["encoder_output", "r_att_cache", "r_cnn_cache"]
torch.onnx.export(asr, dummy_inputs, "fastu2++.onnx", verbose=False, input_names=input_names, output_names=output_names, opset_version=13)