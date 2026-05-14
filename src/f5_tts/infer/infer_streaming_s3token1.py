import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path
import numpy as np
import soundfile as sf
import tomli
from omegaconf import OmegaConf
import torch
import sys
import json
# 打印当前工作目录
# print("当前工作目录:", os.getcwd())
# 获取 f5_tts 的父目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 将父目录添加到模块搜索路径
sys.path.append(project_root)
# 打印 Python 模块搜索路径
print("Python 模块搜索路径:")
for path in sys.path:
    print(f" - {path}")
from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
# from third_party.dspgan.test_vocoder import decode_mel,decode_mel_streaming
from third_party.hifigan.models import Generator
from f5_tts.model import DiT, UNetT
import yaml
import torchaudio
from f5_tts.model.mel_processing import mel_spectrogram_torch_aslp

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

config_path = "/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/configs/fm_10ms_cosyvoice1.yaml"
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
with open(config_path, 'r') as file:  
    config = yaml.safe_load(file) 

model = "F5-TTS"
model_cfg = config.get("model_cfg", str(files("f5_tts").joinpath("configs/fm_10ms_cosyvoice1.yaml")))
model_cfg = config.get("model_cfg", "/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/configs/fm_10ms_cosyvoice1.yaml")
ckpt_file = config.get("ckpt_file", "")
vocab_file = config.get("vocab_file", "")

if model == "F5-TTS":
    model_cls = DiT
    model_cfg = OmegaConf.load(model_cfg).model.arch
    ckpt_file = "/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_10ms_dspgancosyvoice1/model_50000.pt"
    # ckpt_file = "/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_10ms_dspgancosyvoice1/model_100000.pt"
    # ckpt_file = "/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_10ms_dspgancosyvoice1/model_150000.pt"
    ckpt_file = "/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_10ms_dspgancosyvoice1/model_200000.pt"
    ckpt_file = "/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_10ms_dspgancosyvoice1/model_300000.pt"
    # ckpt_file = "/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_10ms_dspgancosyvoice1/model_400000.pt"
# 假设 ema_model 已经定义并加载
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
print(f"Using {model}...")
ema_model = load_model(model_cls, model_cfg,ckpt_file,mel_spec_type="dspgan", vocab_file=vocab_file, vocab_size=4096)

#vocoder
vocoder_config = "/home/node57_data/hkxie/4O/F5-TTS/src/third_party/hifigan/config_streamfm10ms.json"
with open(vocoder_config) as f:
    data = f.read() 
global h
json_config = json.loads(data)
h = AttrDict(json_config)
generator = Generator(h).to(device)
state_dict_g = load_checkpoint("/home/node57_data/hkxie/4O/F5-TTS/src/third_party/hifigan/ckpt_hifigan/g_00300000", device)
generator.load_state_dict(state_dict_g['generator'])

generator.eval()
generator.remove_weight_norm()

## 推理数据
token_path = "/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvocie1_test_token/shoushandashu01.hubert_code.npy"
token_path = "/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvocie1_test_token/kid1.hubert_code.npy"
token_path = "/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvocie1_test_token/ICAGC_8-3-12.hubert_code.npy"
wav_path = "/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test/DB_TTS_0412_000185.wav"
wav_path = "/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test/ICAGC_8-3-12.wav"
wav_path = "/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test/shoushandashu01.wav"
wav_path = "/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test/kid1.wav"
wav_path = "/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test/ICAGC_8-3-12.wav"

token = np.load(token_path)
token = torch.from_numpy(token).unsqueeze(0)  # 转换为 torch.Tensor [batch,token_seq]

audio , sample_rate = torchaudio.load(wav_path)
mel_spec = mel_spectrogram_torch_aslp(y=audio, n_fft=1024, num_mels=80, sampling_rate=16000, hop_size=160, win_size=640, fmin=0, fmax=8000, center=False)
mel_spec = mel_spec.permute(0,2,1)
import pdb;pdb.set_trace()
# mel_spec = mel_spec.squeeze(0)

ref_audio_len = [int(len(token[0])*4)] #[d,t]
output_wav_path = os.path.join("/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/s3token1_fm_streaming",f"{os.path.basename(token_path)[:-4]}.wav")
with torch.inference_mode():

    generated = ema_model.sample_streaming(
        cond=mel_spec,
        text=token,
        duration=ref_audio_len,
        steps=nfe_step,
        cfg_strength=cfg_strength,
        vocoder = generator,
        sway_sampling_coef=sway_sampling_coef,
        output_wav_path = output_wav_path
    )
    generated, _ = ema_model.sample(
        cond=mel_spec,
        text=token,
        duration=ref_audio_len,
        steps=nfe_step,
        cfg_strength=cfg_strength,
        vocoder = generator,
        sway_sampling_coef=sway_sampling_coef,
        output_wav_path = output_wav_path
    )
    
    generated = generated.to(torch.float32)
    gen_mel_spec = generated[:, :, :].permute(0, 2, 1).cpu()

    output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated.wav')
    write(output_file, h.loss_sampling_rate, audio)
    
    with torch.no_grad():
        # decode_mel(gen_mel_spec,f"/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/test/streaming_gen400k_test_1_chunk40.wav")
        decode_mel(gen_mel_spec,"/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/s3token2_fm_steaming/nons_DB_TTS_0412_000185.wav")
        
        # decode_mel(ref_mel_spec,f"{log_samples_path}/update_{global_update}_ref.wav")


