import json
from importlib.resources import files
import os
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from f5_tts.model.ecapa_tdnn import ECAPA_TDNN,ConditioningEncoder
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default
import numpy as np
import random
from f5_tts.model.mel_processing import mel_spectrogram_torch_aslp
import onnxruntime

# cosyvoice spk pretrain extracted_embedding 

class OnlineDataset(Dataset):
    def __init__(
        self,
        data_path,  # 直接从 `1whtraindataset.txt` 读取数据
        target_sample_rate=16000,
        hop_length=200,
        n_mel_channels=80,
        n_fft=1024,
        win_length=800,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = self.load_data(data_path)  # 读取数据文件并解析
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            # 保存参数，不直接调用函数
            self.mel_params = {
                "n_fft": n_fft,
                "num_mels": n_mel_channels,
                "sampling_rate": target_sample_rate,
                "hop_size": hop_length,
                "win_size": win_length,
                "fmin": 0,
                "fmax": 8000,
                "center": False
            }
            
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession("/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/model/campplus.onnx", sess_options=option, providers=["CPUExecutionProvider"])
    def load_data(self, file_path):
        """ 从 txt 文件加载数据 """
        data = []
        dataset_path = os.path.join("/home/node60_tmpdata/hkxie/workspace/streamingfm/data",file_path)
        with open(os.path.join("/home/node60_tmpdata/hkxie/workspace/streamingfm/data",file_path), 'r', encoding='utf-8') as f:
            # print(f"dataset路径为= {dataset_path}")
            for line in f:
                parts = line.strip().split(" ")  # 使用制表符分割
                # print(parts)
                if len(parts) < 4:
                    print(f"数据不足4列")
                    continue  # 确保有足够的列
                utt_id, audio_path, text_token_path, duration = parts
                duration = float(duration)  # 确保 duration 是 float 类型
                data.append({
                    "utt_id": utt_id,
                    "audio_path": audio_path,
                    "text_token_path": text_token_path,
                    "duration": duration
                })
        return data

    def _extract_spk_embedding(self,speech): #campplus_model str查看一下
        feat = kaldi.fbank(speech,
                            num_mel_bins=80,
                            dither=0,
                            sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,{self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).cpu().detach()#.to(self.device) #[bs,192]
        return embedding
    
    def get_frame_len(self, index):
        """ 计算帧长度 """
        duration = min(self.data[index]["duration"], 8.0)
        
        return duration * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text_path = row["text_token_path"]
            duration = row["duration"]
            
            # 过滤时长范围
            if 2.0 <= duration <= 60.0: 
                break  # 满足要求，继续处理

            index = (index + 1) % len(self.data)  # 防止超出索引

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            spk_emb = self._extract_spk_embedding(audio) #可能需要截短？，部分audio可能时长太长
            # print(spk_emb.shape)
            # to mel spectrogram
            # 动态调用函数并传入 y=audio
            mel_spec = mel_spectrogram_torch_aslp(
                y=audio,
                **self.mel_params
            )
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t' #[80,seq_t]
        
        # 读取文本 token
        text_token = np.load(text_path)  # 加载 .hubert_code.npy
        text_token = torch.tensor(text_token, dtype=torch.float32)  # 转为 PyTorch tensor
        
        token_seq_len = len(text_token);upsample_rate=4
        mel_seq_len = mel_spec.shape[1];target_frames=200  # 获取当前 token 的总帧数 卡到8s，batch2
        if token_seq_len > target_frames: #(token,)
            # 随机选择起始帧（避免超出边界）
            start_idx = random.randint(0, token_seq_len - target_frames)
            text_token = text_token[start_idx : start_idx + target_frames,]
            mel_spec = mel_spec[:, start_idx*upsample_rate : (start_idx + target_frames)*upsample_rate]  # 取出 1200 帧
            duration = float(target_frames/25)
                
        return {
            "mel_spec": mel_spec,
            "text": text_token,
            "duration": duration,
            "spk_emb":spk_emb,
        }


class CustomDataset(Dataset):
    def __init__(
        self,
        data_path,  # 直接从 `1whtraindataset.txt` 读取数据
        target_sample_rate=16000,
        hop_length=200,
        n_mel_channels=80,
        n_fft=1024,
        win_length=800,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = self.load_data(data_path)  # 读取数据文件并解析
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = mel_spec_module or MelSpec(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mel_channels=n_mel_channels,
                target_sample_rate=target_sample_rate,
                mel_spec_type=mel_spec_type,
            )

    def load_data(self, file_path):
        """ 从 txt 文件加载数据 """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split("\t")  # 使用制表符分割
                if len(parts) < 4:
                    continue  # 确保有足够的列
                utt_id, mel_path, text_token_path, duration = parts
                duration = float(duration)  # 确保 duration 是 float 类型
                data.append({
                    "utt_id": utt_id,
                    "mel_path": mel_path,
                    "text_token_path": text_token_path,
                    "duration": duration
                })
        return data

    def get_frame_len(self, index):
        """ 计算帧长度 """
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            mel_path = row["mel_path"]
            text_path = row["text_token_path"]
            duration = row["duration"]
            
            # 过滤时长范围
            if 1.3 <= duration <= 15.0: 
                break  # 满足要求，继续处理

            index = (index + 1) % len(self.data)  # 防止超出索引

        # 读取 mel 特征
        mel_spec = np.load(mel_path)  # 加载 mel.npy 文件
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32)  # 变成 PyTorch tensor
        # mel_spec = 
        
        # 读取文本 token
        text_token = np.load(text_path)  # 加载 .hubert_code.npy
        text_token = torch.tensor(text_token, dtype=torch.long)  # 转为 PyTorch tensor

        return {
            "mel_spec": mel_spec,
            "text": text_token,
            "duration": duration,
        }

# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_last: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        # import pdb;pdb.set_trace()
        indices.sort(key=lambda elem: elem[1])
        
        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset

def load_dataset(
    dataset_path: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """
    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        print(f"使用{dataset_type}")
        train_dataset = CustomDataset(
            data_path=dataset_path,
            preprocessed_mel=True,  # 这里假设 mel 已经预处理，若没有需要修改
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )
        return train_dataset
    elif dataset_type == "OnlineDataset":
        print(f"使用{dataset_type}")
        train_dataset = OnlineDataset(
            data_path=dataset_path,
            preprocessed_mel=False,  # 这里假设 mel 需要预处理
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )
        return train_dataset
# /home/node57_data/hkxie/4O/streaming_fm/dataset/1whtraindataset.txt


# collation

def collate_fn(batch):
    # batch 是一个 list，包含多个样本，每个样本是一个 dict，mel_spec 是 (mel_dim, seq_len)
    mel_specs = [item["mel_spec"] for item in batch]  # List of [dim=mel_dim, seq]

    bs = len(mel_specs)  # 获取 batch 大小
    mel_dim = mel_specs[0].shape[0]  # Mel 维度
    target_frames = 300  # 3 秒的帧数 300*10ms
    
    ref_mels = []
    for mel_spec in mel_specs:
        seq_len = mel_spec.shape[1]  # 获取当前 mel 的总帧数
        if seq_len < target_frames:
            # 如果 mel_spec 长度小于 target_frames，进行 padding
            padding = (0, target_frames - seq_len)  # 在时间维度上 padding
            padded_mel_spec = F.pad(mel_spec, padding, mode='constant', value=0)  # 使用 0 填充
            ref_mels.append(padded_mel_spec)
        else:
            # 随机选择起始帧（避免超出边界）
            start_idx = random.randint(0, seq_len - target_frames)
            ref_mel = mel_spec[:, start_idx : start_idx + target_frames]  # 取出 375 帧
            ref_mels.append(ref_mel)

    # # 确保至少有一个 ref_mel
    # if len(ref_mels) == 0:
    #     raise ValueError("所有样本的 mel 长度都不足 3s，请检查数据！")
    
    # # 如果 ref_mels 数量不足 batch_size，则随机重复已有样本
    # while len(ref_mels) < bs:
    #     ref_mels.append(random.choice(ref_mels))  # 从已有的 ref_mels 随机选一个重复

    # 拼接成 batch 维度 [bs, mel_dim, 375]
    ref_mels = torch.stack(ref_mels)
    
    #交换维度，变为 [bs, 375, mel_dim]
    ref_mels = ref_mels.permute(0, 2, 1)
    # # 送入模型
    # model = ECAPA_TDNN(in_channels=mel_dim, channels=512, embd_dim=128)

    # # model = ConditioningEncoder(spec_dim=mel_dim,embedding_dim=128)
    
    # ref_embed = model(ref_mels)  # [bs, embd_dim]
    # ref_embed = ref_embed.cpu().detach()
    
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    #pad_sequence 已经返回一个堆叠的张量，因此 stack 是多余的 #0填充？asr_emb
    text = pad_sequence(text, padding_value=0, batch_first=True)
    # text = torch.stack(text)
    text_lengths = torch.LongTensor([len(item) for item in text])
    
    spk_emb = [item["spk_emb"] for item in batch]
    
    spk_emb = torch.stack(spk_emb)
    # print("spk_emb.shape=",spk_emb.shape)
    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
        ref_embed = ref_mels,
        spk_emb = spk_emb,
    )
