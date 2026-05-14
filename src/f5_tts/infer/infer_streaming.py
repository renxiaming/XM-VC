import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from omegaconf import OmegaConf
import torch
import sys
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
from third_party.dspgan.test_vocoder import decode_mel,decode_mel_streaming
from f5_tts.model import DiT, UNetT
import hydra
import yaml
import argparse
parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
args = parser.parse_args()

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5-TTS | E2-TTS",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))

config_path = "/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/configs/fm_Base_train.yaml"
# cfg = load_hparams_from_json(config_path)
# config = tomli.load(open(config_path, "rb"))

with open(config_path, 'r') as file:  
    config = yaml.safe_load(file) 


# command-line interface parameters

# model = args.model or config.get("model", "F5-TTS")
# model_cfg = args.model_cfg or config.get("model_cfg", str(files("f5_tts").joinpath("configs/fm_Base_train.yaml")))
# ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
# vocab_file = args.vocab_file or config.get("vocab_file", "")

model = "F5-TTS"
model_cfg = config.get("model_cfg", str(files("f5_tts").joinpath("configs/fm_Base_train.yaml")))
ckpt_file = config.get("ckpt_file", "")
vocab_file = config.get("vocab_file", "")

# model_cls = DiT

# model = CFM(
#     transformer=model_cls(**cfg.model.arch, text_num_embeds=cfg.model.text_num_embeds, mel_dim=cfg.model.mel_spec.n_mel_channels),
#     mel_spec_kwargs=cfg.model.mel_spec,
# )

if model == "F5-TTS":
    model_cls = DiT
    
    model_cfg = OmegaConf.load(model_cfg).model.arch
    ckpt_file = "/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_Base_dspgan_pinyin_/home/node57_data/hkxie/4O/streaming_fm/dataset/1whtraindataset.txt/model_50000.pt"
    # ckpt_file = "/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_Base_dspgan_pinyin_/home/node57_data/hkxie/4O/streaming_fm/dataset/1whtraindataset.txt/model_100000.pt"
    # ckpt_file = "/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_Base_dspgan_pinyin_/home/node57_data/hkxie/4O/streaming_fm/dataset/1whtraindataset.txt/model_150000.pt"
    ckpt_file = "/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_Base_dspgan_pinyin_/home/node57_data/hkxie/4O/streaming_fm/dataset/1whtraindataset.txt/model_200000.pt"
    ckpt_file = "/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_Base_dspgan_pinyin_/home/node57_data/hkxie/4O/streaming_fm/dataset/1whtraindataset.txt/model_300000.pt"
    ckpt_file = "/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_Base_dspgan_pinyin_/home/node57_data/hkxie/4O/streaming_fm/dataset/1whtraindataset.txt/model_400000.pt"
# 假设 ema_model 已经定义并加载
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
print(f"Using {model}...")
ema_model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type="dspgan", vocab_file=vocab_file)


# 将模型移动到指定设备
# ema_model.to(device)

text = "而且那时候我上学的时候成绩也不差，我都，我都会，但是我就是说，唉呀，这个我都忘了，我曾经学过这个怎么读的呀。"
# CUDA_VISIBLE_DEVICES=1 python3 infer.py
token_path = "/home/node57_data/hkxie/4O/streaming_fm/data/s3token2/05343304771_EIjYa_VAD41_6.hubert_code.npy"
# token_path = "/home/node57_data/hkxie/dataset/t2s_final/val/00675_00676_0310_2.hubert_code.npy"
# token_path = "/home/node57_data/hkxie/dataset/t2s_0113/test/00615_00616_0099_2.hubert_code.npy"
token_path = "/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test/DB_TTS_0412_000185.hubert_code.npy"
ref_path = "/home/work_nfs14/dkguo/TTS_Testset/lm/promptset/hq_set/mels/DB_TTS_0412_000185.npy"
# ref_path = "/home/work_nfs14/dkguo/TTS_Testset/lm/promptset/hq_set/mels/DB_TTS_0412_000183.npy"
# ref_path = "/home/work_nfs14/dkguo/TTS_Testset/lm/promptset/hq_set/mels/DB_TTS_0412_000206.npy"

token = np.load(token_path)
ref = np.load(ref_path)

# 将 numpy 数组转换为 PyTorch 张量
token = torch.from_numpy(token).unsqueeze(0)  # 转换为 torch.Tensor [batch,token_seq]
ref = torch.from_numpy(ref)     # 转换为 torch.Tensor
ref = ref.unsqueeze(0).permute(0,2,1)
# import pdb;pdb.set_trace()

ref_audio_len = [int(len(token[0])*3.2)] #[d,t]
output_wav_path = os.path.join("/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/s3token2_fm_steaming",f"{os.path.basename(ref_path)[:-4]}.wav")
with torch.inference_mode():

    generated = ema_model.sample_streaming(
        cond=ref,
        text=token,
        duration=ref_audio_len,
        steps=nfe_step,
        cfg_strength=cfg_strength,
        vocoder = decode_mel_streaming,
        sway_sampling_coef=sway_sampling_coef,
        output_wav_path = output_wav_path
    )
    generated, _ = ema_model.sample(
        cond=ref,
        text=token,
        duration=ref_audio_len,
        steps=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
    )
    generated = generated.to(torch.float32)
    gen_mel_spec = generated[:, :, :].permute(0, 2, 1).cpu()
    # ref_mel_spec = batch["mel"][0].unsqueeze(0)
    
    with torch.no_grad():
        # decode_mel(gen_mel_spec,f"/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/test/streaming_gen400k_test_1_chunk40.wav")
        decode_mel(gen_mel_spec,"/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/s3token2_fm_steaming/nons_DB_TTS_0412_000185.wav")
        
        # decode_mel(ref_mel_spec,f"{log_samples_path}/update_{global_update}_ref.wav")


