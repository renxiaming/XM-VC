import time
import numpy as np
# import pyaudio
from tqdm import tqdm
import torch
from threading import Thread, Lock
import torchaudio.compliance.kaldi as kaldi
import librosa
torch.set_num_threads(1)

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
    fbanks = fbanks.unsqueeze(0)
    return fbanks

class VCRunner():
    def __init__(self, spkid):
        self.mutex = Lock()
        torch.set_num_threads(1)
        self.asr = torch.jit.load('models/fastu2++.pt')
        self.vc = torch.jit.load('models/vc_fuxiall.pt')
        self.vocoder = torch.jit.load('models/refineganft.pt')

        decoding_chunk_size = 5
        num_decoding_left_chunks = 2
        subsampling = 4
        context = 7  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        self.required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        self.CHUNK = 160 * stride
        self.vc_chunk = int(decoding_chunk_size * 16 / 5)
        self.vocoder_overlap = 2
        upsample_factor = 200
        self.vocoder_wav_overlap = self.vocoder_overlap * upsample_factor

        self.down_linspace = torch.linspace(1, 0, steps=self.vocoder_wav_overlap, out=None).numpy()
        self.up_linspace = torch.linspace(0, 1, steps=self.vocoder_wav_overlap, out=None).numpy()

        # p = pyaudio.PyAudio()
        # self.in_stream = p.open(format=pyaudio.paInt16,
        #                 channels=1,
        #                 rate=16000,
        #                 input=True,
        #                 frames_per_buffer=self.CHUNK)
        #self.out_stream = p.open(format=pyaudio.paFloat32, channels=1, rate=48000, output=True, output_device_index=3)
        #self.out_stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, output=True)

        #self.spks = torch.LongTensor([spkid])
        self.spks = torch.from_numpy(np.load('../testdata/xvectors/SSB06860172.npy')).unsqueeze(0)
        #self.spks = torch.from_numpy(np.load('xunmengchu.npy')).unsqueeze(0)
        #self.spks = torch.from_numpy(np.load('maomao_0027.npy')).unsqueeze(0)
        #self.spks = torch.from_numpy(np.load('0036.npy')).unsqueeze(0)
        #self.spks = torch.from_numpy(np.load('hemoyi_0020.npy')).unsqueeze(0)
        #self.spks = torch.from_numpy(np.load('renwuxian_0004.npy')).unsqueeze(0)
        #self.spks = torch.from_numpy(np.load('000001.npy')).unsqueeze(0)



    def playaudio(self, out_stream, data):
        with self.mutex:
            out_stream.write(data)

    def init_cache(self):
        self.samples_cache_len = 240 + 3 * 160
        self.samples_cache = np.ones(self.samples_cache_len) * -0.5

        self.att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device='cpu')
        self.cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device='cpu')
        self.asr_offset = 10
        self.encoder_output_cache = None

        self.vc_offset = 32
        self.vc_cache = None

        self.vocoder_cache = None
        self.last_wav = None
    
    def reset_cache(self):
        self.asr_offset = 10
        self.vc_offset = 32


    def inference_one_chunk(self, samples):
        with torch.no_grad():
            samples = np.concatenate((self.samples_cache, samples))
            self.samples_cache = samples[-self.samples_cache_len:]
            fbanks = extract_fbanks(samples, frame_shift=10).float()
            (encoder_output, self.att_cache, self.cnn_cache) = self.asr.forward_encoder_chunk(
                fbanks, self.asr_offset, self.required_cache_size, self.att_cache, self.cnn_cache)
            #print(fbanks.shape, self.required_cache_size, self.att_cache.shape, self.cnn_cache.shape, encoder_output.shape)

            self.asr_offset += encoder_output.size(1)
            if self.encoder_output_cache is None:
                encoder_output = torch.cat([encoder_output[:, 0:1, :], encoder_output], dim=1)
            else:
                encoder_output = torch.cat([self.encoder_output_cache, encoder_output], dim=1)
            self.encoder_output_cache = encoder_output[:, -1:, :]
            encoder_output_upsample = encoder_output.transpose(1, 2)
            encoder_output_upsample = torch.nn.functional.interpolate(encoder_output_upsample, size=self.vc_chunk + 1, mode='linear', align_corners=True)
            encoder_output_upsample = encoder_output_upsample.transpose(1, 2)
            encoder_output_upsample = encoder_output_upsample[:, 1:, :]



            generate_output_chunk, self.vc_cache = self.vc.inference(
                encoder_output_upsample,
                self.spks,
                self.vc_offset,
                0,
                cache=self.vc_cache,
            )
            #print(encoder_output_upsample.shape, self.spks.shape)
            self.vc_offset += self.vc_chunk

            mel = generate_output_chunk.transpose(1, 2)

            if self.vocoder_cache is not None:
                mel = torch.cat([self.vocoder_cache, mel], dim=-1)
            self.vocoder_cache = mel[:, :, -self.vocoder_overlap:]
            #mel = mel / 4.
            wav = self.vocoder.inference(mel).squeeze()
            wav = wav.detach().cpu().numpy()

            wav = librosa.resample(wav, orig_sr=48000, target_sr=16000)
            
            if self.last_wav is not None:
                front_wav = wav[:self.vocoder_wav_overlap]
                smooth_front_wav = self.last_wav * self.down_linspace + front_wav * self.up_linspace
                new_wav = np.concatenate([smooth_front_wav, wav[self.vocoder_wav_overlap:-self.vocoder_wav_overlap]], axis=0)
            else:
                new_wav = wav[:-self.vocoder_wav_overlap]
            self.last_wav = wav[-self.vocoder_wav_overlap:]

            return new_wav

    def run(self):
        # p = pyaudio.PyAudio()
        # info = p.get_host_api_info_by_index(0)
        # numdevices = info.get('deviceCount')

        # for i in range(0, numdevices):
        #     if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        #         print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
        #device_id = int(input("device id:"))
        # device_id = 3
        # if device_id < 0:
        #     device_id = None
        print("warming up")
        self.init_cache()
        for i in tqdm(range(10)):
            # data = self.in_stream.read(self.CHUNK, exception_on_overflow = False)
            # samples = np.fromstring(data, dtype='float32')
            # samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (1 << 15)
            shape = shape = (3200, )
            samples = np.random.rand(*shape).astype(np.float32)
            self.inference_one_chunk(samples)
        # self.in_stream = p.open(format=pyaudio.paInt16,
        #                 channels=1,
        #                 rate=16000,
        #                 input=True,
        #                 frames_per_buffer=self.CHUNK)
        # self.out_stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, output=True, output_device_index=device_id)
        #self.out_stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, output=True)
        # for i in tqdm(range(10)):
        #     data = self.in_stream.read(self.CHUNK, exception_on_overflow = False)
        #     samples = np.fromstring(data, dtype='float32')
        #     samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (1 << 15)
            # Thread(target=self.playaudio, args=(self.out_stream, samples.tobytes())).start()
        i = 0
        spk_cnt = 0
        while True:
            #print("running...")
            #if i % 100 * 20 == 0:
            if i % 200 == 0:
                print("reset!")
                self.reset_cache()
                #spk_cnt = (spk_cnt) % 44
                #self.spks = torch.LongTensor([spk_cnt])
                #spk_cnt += 1
            # data = self.in_stream.read(self.CHUNK, exception_on_overflow = False)
            # samples = np.fromstring(data, dtype='float32')
            # samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (1 << 15)
            shape = shape = (3200, )
            samples = np.random.rand(*shape).astype(np.float32)
            cur_time = time.time()
            syn_wav = self.inference_one_chunk(samples)
            print(f"chunk use time{time.time()-cur_time}")
            # Thread(target=self.playaudio, args=(self.out_stream, syn_wav.tobytes())).start()
            i += 1




vc = VCRunner(0)
vc.run()
