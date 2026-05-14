"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random
import random
from typing import Callable
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
import numpy as np
from scipy.io.wavfile import write
from f5_tts.model.modules import MelSpec
from f5_tts.model.ecapa_tdnn import ECAPA_TDNN,ConditioningEncoder
from f5_tts.model.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


def final_convert_wav(wav_data):
    audio = wav_data.data.cpu().float().numpy()

    audio = audio * 32768.0
    audio = audio.astype(np.int16)
    return audio

class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()
        # import pdb;pdb.set_trace()
        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # import pdb;pdb.set_trace()
        # raw wave
        self.text = text
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels
        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        cond = cond.to(next(self.parameters()).dtype) #["b n d"]
        bs,seq_len,mel_dim = cond.shape
        target_frames = 3*mel_dim  # 3 秒的帧数
        model = ConditioningEncoder(spec_dim=80,embedding_dim=128)
        ref_mels = []
        if seq_len < target_frames:
            # 如果 mel_spec 长度小于 target_frames，进行 padding
            padding = (0, target_frames - seq_len)  # 在时间维度上 padding
            ref_mel = F.pad(cond, padding, mode='constant', value=0)  # 使用 0 填充
            ref_mels.append(ref_mel)
        else:
            # 随机选择起始帧（避免超出边界）
            start_idx = random.randint(0, seq_len - target_frames)
            ref_mel = cond[:, start_idx : start_idx + target_frames]  # 取出 375 帧
            ref_mels.append(ref_mel)
        
        # 拼接成 batch 维度 [bs, 80, 375]
        ref_mels = torch.stack(ref_mels)
        ref_mels = ref_mels.squeeze(0).permute(0,2,1).float()
        ref_embed = model(ref_mels)  # [bs, embd_dim]
        ref_embed = ref_embed.half().detach().to(device)
        cond = ref_embed
        self.cond = cond
        
        # #y0 3000 diyige chunk 第一次 chunk_size + future
        #             第二次 past + chunk_size + future
        #             token 也截取对应窗口，每一个都过一次 sample
        #             block sample 
        #             3.2 可能需要配平，upsample chunk_size找个合适的。chunk_size*3.2=整数？
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        #cond截断
        
        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None
        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
            # import pdb;pdb.set_trace()
            # predict flow
            device = self.device  # 统一到 self.device 
            x = x.to(device);t = t.to(device);cond = self.cond.to(device);text = self.text.to(device)
            pred = self.transformer(
                x=x, cond=cond, text=text, time=t, mask=mask, drop_audio_cond=False, drop_text=False
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=cond, text=text, time=t, mask=mask, drop_audio_cond=True, drop_text=True
            )
            return pred + (pred - null_pred) * cfg_strength

        
        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(int(dur), self.num_channels, device=self.device, dtype=cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        
        # import pdb;pdb.set_trace()
        sampled = trajectory[-1]
        out = sampled
        # out = torch.where(cond_mask, cond, out)
        
        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def sample_streaming(
        self,
        cond: torch.FloatTensor,      # [b, n, d] or [b, nw]
        text: torch.LongTensor | list[str],
        duration: int,
        *,
        lens: torch.Tensor | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        block_size=25,       # 每块帧数
        past_blocks=1,
        current_blocks=2,
        future_blocks=1,
    ):
        """
        以块为单位进行流式采样的示例逻辑。
        - past_blocks=1, current_blocks=2, future_blocks=1
        - 每个block_size=25 帧
        - 总窗口大小 = (past_blocks + current_blocks + future_blocks) * block_size
        - 每次只输出 current_blocks * block_size 的部分
        - 然后窗口向后滑动 current_blocks 个 block
        """

        self.eval()
        device = cond.device
        # 1) 先做一次性的参考 embedding（跟原先逻辑一致）
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len = cond.shape[:2]
        cond = cond.to(next(self.parameters()).dtype)
        bs, seq_len, mel_dim = cond.shape

        # 示例：构造参考 embedding
        model = ConditioningEncoder(spec_dim=80, embedding_dim=128)
        # 简单处理：如果 seq_len 不足，就 padding，否则随机取 3秒
        target_frames = 3 * mel_dim
        if seq_len < target_frames:
            padding = (0, target_frames - seq_len)
            ref_mel = F.pad(cond, padding, mode='constant', value=0)
        else:
            start_idx = random.randint(0, seq_len - target_frames)
            ref_mel = cond[:, start_idx : start_idx + target_frames]

        ref_mel = ref_mel.permute(0,2,1).float()  # [b, 80, T]
        ref_embed = model(ref_mel).half().detach().to(self.device)
        self.cond = ref_embed
        self.text = text #[batch,token_seq]

        # 如果没有 lens，就全用
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        if batch > 1:
            mask = lens_to_mask(lens)
        else:
            mask = None

        # 定义 Flow Matching 的预测函数
        def fn(t, x):#, cond_, text_, mask_
            """
            这里将 cond_、text_、mask_ 作为参数传入，以便块内调用。
            """
            
            device = self.device
            x = x.to(device)
            t = t.to(device)
            cond_ = self.cond.to(device)
            
            # # if isinstance(text_, torch.Tensor):
            text_ = text_window.to(device)

            pred = self.transformer(
                x=x, cond=cond_, text=text_, time=t, mask=mask,
                drop_audio_cond=False, drop_text=False,chunk_offset=frame_left
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=cond_, text=text_, time=t, mask=mask,
                drop_audio_cond=True, drop_text=True,chunk_offset=frame_left
            )
            return pred + (pred - null_pred) * cfg_strength

        # 2) 准备整段初始噪声 y0: 假设 total_duration 为整段长度
        #    这里仅示例 batch=1 的情况，多 batch 需再行扩展
        # print(duration)
        # if isinstance(duration, int):
            # total_duration = [duration]
        # total_duration = [duration]
        y0_full = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
                
            y0_full.append(torch.randn(int(dur), self.num_channels, device=self.device, dtype=ref_embed.dtype))
        y0_full = pad_sequence(y0_full, padding_value=0, batch_first=True)  # [b, max_dur, num_channels]

        # 3) 分块相关参数
        window_blocks = past_blocks + current_blocks + future_blocks # 1 2 1
        window_size = window_blocks * block_size   # 例如 4 * 25 = 100
        shift_size = current_blocks * block_size   # 每次滑动 2 * 25 = 50
        
        # 计算总块数
        # 假设 total_dur = y0_full.shape[1] (即实际有效时长)
        total_dur = y0_full.shape[1]
        # 以 chunk_size 为粒度，算出最大块数
        total_num_blocks = math.ceil(total_dur / block_size)
        
        # 4) 逐块滑动采样
        generated_chunks = []
        generated_wavs = []
        current_start_block = 0

        while True:
            # 当前窗口左边界 = current_start_block - past_blocks
            # 右边界 = current_start_block + current_blocks + future_blocks - 1
            past_start = current_start_block - past_blocks
            current_end = current_start_block + current_blocks - 1
            future_end = current_start_block + current_blocks + future_blocks - 1

            # 转换成帧级别的 index
            frame_left = past_start * block_size
            frame_right = (future_end + 1) * block_size  # 右边界是闭区间，所以 +1

            # 边界裁剪
            frame_left = max(0, frame_left)
            frame_right = min(total_dur, frame_right)

            if frame_left >= total_dur:
                # 已经没有可生成的内容
                break
            # import pdb;pdb.set_trace()
            # 截取噪声 y0_window: [b, window_frames, num_channels]
            y0_window = y0_full[:, frame_left:frame_right, :]

            # 截取对应的 text token
            text_window = self.text[:,int(frame_left/3.2):int(frame_right/3.2)] #[batch,token_seq]

            # 若窗口长度不足 window_size，则可以做 padding；这里示例直接 padding 到 window_size
            actual_window_len = y0_window.shape[1]
            if actual_window_len < window_size:
                pad_len = window_size - actual_window_len
                y0_window = F.pad(y0_window, (0, 0, 0, pad_len), mode='constant', value=0)

            # 准备时间步
            t = torch.linspace(0, 1, steps + 1, device=self.device, dtype=ref_embed.dtype)
            if sway_sampling_coef is not None:
                t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            # t = t.to(self.device)
            # y0_window = y0_window.to(self.device)
            
            # ODE 积分
            trajectory = odeint(
                fn,
                y0_window,
                t,
                **self.odeint_kwargs
            )
            # trajectory = odeint(
            #     lambda _t, _x: fn(_t, _x, self.cond, text_window, mask),
            #     y0_window,
            #     t,
            #     **self.odeint_kwargs
            # )
            sampled_window = trajectory[-1]  # [b, window_size, num_channels]

            # 只取其中的 "current" 部分输出
            # current block 范围 = [current_start_block, current_start_block + current_blocks-1]
            # 换成帧下标 = [current_start_block * block_size, (current_start_block+current_blocks)*block_size)
            current_frame_left = current_start_block * block_size
            current_frame_right = (current_start_block + current_blocks) * block_size

            # 但是在 sampled_window 里，0 对应 frame_left
            # 所以需要把 current_frame_left 映射到窗口内:
            local_current_left = (current_frame_left - frame_left)
            local_current_right = (current_frame_right - frame_left)

            # 裁剪
            current_part = sampled_window[:, local_current_left:local_current_right, :]  # [b, current_blocks*block_size, num_channels]
            generated_chunks.append(current_part)
            
            # current_wav = vocoder(current_part.permute(0, 2, 1).float().cpu()) #float32 cpu
            
            #先整体过vocoder合成，然后再截取中间cache下来 
            upsample_rate = int(48000/80)# sample_rate // mel_dim
            window_wav = vocoder(sampled_window.permute(0, 2, 1).float().cpu())
            # import pdb;pdb.set_trace()
            current_wav = window_wav[local_current_left*upsample_rate:local_current_right*upsample_rate]
            generated_wavs.append(current_wav) #每个wav长度等于 sample rate * 2*chunk/80dim
            # final_convert_wav
            # 滑动到下一个
            current_start_block += current_blocks
            if current_frame_right >= total_dur:
                # 已经到达末尾
                break
        import pdb;pdb.set_trace()
        out_wav = np.concatenate(generated_wavs)
        import time
        write(f"test_streaming_block{block_size}{int(time.time())}.wav", 48000, out_wav)
        # 拼接所有的 current 部分
        out = torch.cat(generated_chunks, dim=1)  # [b, total_generated_frames, num_channels]
        
        # 5) 若有 vocoder，则做后处理
        # if exists(vocoder):
        #     # vocoder 可能需要 [b, n, d]，也可能是 [b, d, n]，根据实际情况做 permute
        #     out = out.permute(0, 2, 1)  # [b, num_channels, T]
        #     out = vocoder(out)

        yield out,trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
        ref_embed,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma
        # import pdb;pdb.set_trace()
        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # # get a random span to mask out for training conditionally
        # frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        # rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        # if exists(mask):
        #     rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step 
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        ##改为lognorm(0,1) #sigmoid(0,1) #
        # time_np = np.random.lognormal(0, 1, size=(batch,))
        # time_np = (time_np - time_np.min()) / (time_np.max() - time_np.min() + 1e-8)
        # time = torch.tensor(time_np, dtype=dtype, device=self.device)
        
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        #不需要使用mask icl，直接提供cond即可
        # only predict what is within the random mask span for infilling
        # cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
        cond = ref_embed #ref spk mel
        # transformer and cfg training with a drop rate
        drop_audio_cond = random.random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random.random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text
        )
        
        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[mask]

        return loss.mean(), cond, pred

   def streaming_sample(
        self,
        cond: torch.Tensor,       # 条件输入，可以是 waveform 或预先计算的 mel 频谱
        text: torch.Tensor | list[str],  # 文本条件
        total_duration: int,        # 需要生成的总帧数（例如 mel 频谱的时间步数）
        *,
        chunk_size: int = 100,      # 每个生成块的帧数
        pre_context: int = 20,      # 每个块使用的前置上下文帧数（来自上一块的尾部）
        post_context: int = 20,     # 每个块使用的后置上下文帧数（用于平滑过渡，允许轻微延迟）
        steps: int = 32,            # 每个块内部的 ODE 积分步数
        cfg_strength: float = 1.0,  # classifier-free guidance 强度
        seed: int | None = None,    # 随机种子
        vocoder: Callable[[torch.Tensor], torch.Tensor] | None = None,  # vocoder 函数（如 BigVGAN）
    ) -> Iterator[torch.Tensor]:
        """
        流式采样函数：
          1. 首先对 cond 进行预处理，如果是 waveform 则转换为 mel 频谱，
             并通过一个条件编码器获得条件嵌入（参见论文中条件流匹配的设计）。
          2. 将待生成的总帧数分成若干块，每块除当前生成帧外，
             还额外补充 pre_context 和 post_context 帧，用于上下文补全，
             保证各块之间平滑衔接。
          3. 对于每个块，构造初始状态（若有前块生成的尾部则复用，否则随机初始化），
             并采用 RK4 方式在时间网格上进行 ODE 积分，生成当前块的 mel 频谱。
          4. 从积分结果中提取出当前块对应的部分，经过 vocoder 后 yield 波形（或直接 yield mel）。
          
        这种设计正是论文 3.2 节中提出的分块（chunk-by-chunk）生成方法，
        通过滑动窗口和上下文补充，实现低延迟、连续流式的语音生成&#8203;:contentReference[oaicite:1]{index=1}。
        """
        self.eval()
        self.text = text  # 保存文本条件
        
        # --- 1. 条件预处理 ---
        if cond.ndim == 2:
            # 如果输入 cond 为二维，视为 waveform，则先转换为 mel 频谱
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)  # 期望形状：[batch, time, mel_bins]
            assert cond.shape[-1] == self.num_channels, "通道数不匹配"
        batch, cond_seq_len, _ = cond.shape
        cond = cond.to(next(self.parameters()).dtype)
        bs, seq_len, mel_dim = cond.shape
        
        # 采用条件编码器从参考片段获得条件嵌入（参照论文中的 CFM 设计）
        target_frames = 3 * mel_dim  # 参考片段大致对应 3 秒
        cond_encoder = DummyConditioningEncoder(spec_dim=mel_dim, embedding_dim=128).to(cond.device)
        if seq_len < target_frames:
            padding = (0, target_frames - seq_len)
            ref_mel = F.pad(cond, padding, mode='constant', value=0)
        else:
            start_idx = random.randint(0, seq_len - target_frames)
            ref_mel = cond[:, start_idx: start_idx + target_frames]
        # ref_mel 的形状：[batch, target_frames, mel_dim]
        ref_embed = cond_encoder(ref_mel)  # 得到 [batch, embedding_dim]
        ref_embed = ref_embed.half().detach().to(cond.device)
        self.cond = ref_embed
        
        # 对于流式推理，通常 batch = 1
        mask = None  # 如果需要，可构造 mask
        
        # --- 2. 定义 ODE 动力学函数 ---
        def fn(t, x):
            # 此处 x: [batch, time, num_channels]
            x = x.to(self.device)
            t = t.to(self.device)
            cond_local = self.cond.to(self.device)
            text_local = self.text.to(self.device) if torch.is_tensor(self.text) else self.text
            pred = self.transformer(
                x=x, cond=cond_local, text=text_local, time=t, mask=mask,
                drop_audio_cond=False, drop_text=False
            )
            if cfg_strength < 1e-5:
                return pred
            null_pred = self.transformer(
                x=x, cond=cond_local, text=text_local, time=t, mask=mask,
                drop_audio_cond=True, drop_text=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        # --- 3. 分块生成：滑动窗口方式 ---
        n_chunks = ceil(total_duration / chunk_size)
        previous_chunk_tail = None  # 保存上一块输出的尾部，用作当前块的前向上下文
        
        for i in range(n_chunks):
            # 当前块需要生成的帧数（最后一块可能不足 chunk_size）
            current_chunk_frames = chunk_size if (i < n_chunks - 1) else (total_duration - i * chunk_size)
            # 当前块总长度 = pre_context + 当前块 + post_context
            current_block_length = pre_context + current_chunk_frames + post_context
            
            # --- 3.1 准备初始状态 ---
            # 如果有上一块的输出尾部，则作为 pre_context 使用；否则初始化为全零
            if previous_chunk_tail is not None:
                # previous_chunk_tail: [1, pre_context, num_channels]
                pre_noise = previous_chunk_tail
            else:
                pre_noise = torch.zeros(1, pre_context, self.num_channels, device=self.device, dtype=cond.dtype)
            # 当前块和后置上下文部分均随机初始化
            if exists(seed):
                torch.manual_seed(seed + i)  # 保证不同块种子不同
            current_noise = torch.randn(1, current_chunk_frames + post_context, self.num_channels, 
                                        device=self.device, dtype=cond.dtype)
            # 拼接得到完整的初始状态 y0，其形状为 [1, current_block_length, num_channels]
            y0 = torch.cat([pre_noise, current_noise], dim=1)
            
            # --- 3.2 构造时间网格 ---
            t_start = 0.0
            t_grid = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=cond.dtype)
            dt = t_grid[1] - t_grid[0]
            
            # --- 3.3 流式 ODE 积分：逐步更新 ---
            x = y0
            for j in range(steps):
                # x = rk4_step(fn, t_grid[j], dt, x)
                x = odeint(fn, y0, t_grid[j], **self.odeint_kwargs)
            # x 的形状为 [1, current_block_length, num_channels]
            
            # --- 3.4 提取当前块输出 ---
            # 舍去前置和后置上下文，仅保留当前块对应部分
            chunk_output = x[:, pre_context:pre_context + current_chunk_frames, :]
            
            # 更新上一块尾部，用于下一块的 pre_context（取当前块最后 pre_context 帧）
            if pre_context > 0:
                previous_chunk_tail = x[:, pre_context + current_chunk_frames - pre_context: pre_context + current_chunk_frames, :]
            else:
                previous_chunk_tail = None
            
            # --- 3.5 后处理与输出 ---
            if vocoder is not None:
                # 转置为 [batch, num_channels, time] 以适应 vocoder 输入格式
                waveform_chunk = vocoder(chunk_output.permute(0, 2, 1))
                yield waveform_chunk
            else:
                yield chunk_output
