#!/usr/bin/env python3
# Copyright (c) 2021 Binbin Zhang
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Tuple
from typeguard import check_argument_types

from wenet.utils.common import get_activation
from wenet.utils.mask import make_pad_mask, add_optional_chunk_mask
from wenet.cnn.attention import MultiHeadedAttention
from wenet.cnn.positionwise_feed_forward import PositionwiseFeedForward
from wenet.cnn.convolution import ConvolutionModule
from wenet.cnn.encoder_layer import ConformerCnnEncoderLayer
from wenet.cnn.subsampling import Conv2dSubsampling4, Conv2dSubsampling8


class ConformerCnnEncoder(torch.nn.Module):
    """Conformer encoder module. Fully-convolutional version."""
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        cnn_inner_channel: int = 512,
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        activation_type: str = "relu",
        cnn_module_kernel: int = 7,
        causal: bool = False,
        final_norm: str = "layer_norm",
    ):
        """Construct ConformerEncoder

        Args:
            cnn_inner_channel (int): inner channel of conv module.
            final_norm (str): normalization type for final output.

            Other args have the same meaning as in `ConformerEncoder`.
        """
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.global_cmvn = global_cmvn
        activation = get_activation(activation_type)

        # subsampling module definition
        if input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
        )

        # self-attention module definition
        encoder_attn_layer = MultiHeadedAttention
        encoder_attn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
        )

        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (
            output_size,
            cnn_inner_channel,
            cnn_module_kernel,
            activation, causal, True
        )

        self.encoders = torch.nn.ModuleList([
            ConformerCnnEncoderLayer(
                output_size,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args),
                dropout_rate,
            ) for _ in range(num_blocks)
        ])

        self.final_norm = final_norm
        if final_norm == 'layer_norm':
            self.after_norm = torch.nn.LayerNorm(self.output_size(), eps=1e-12)
        else:
            raise ValueError("unknown norm_type: " + final_norm)

        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

    def fuse_modules(self):
        self.embed.fuse_modules()
        for layer in self.encoders:
            layer.fuse_modules()

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: not used.
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        B, T, D = xs.size()
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs = xs.unsqueeze(1)  # (B, C=1, T, D)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, masks = self.embed(xs, masks)
        # xs    (b, size, 1, T/sub_rate)
        # masks (b, 1, T/sub_rate)
        mask_pad = masks.unsqueeze(1).type_as(xs).expand(
            B, 1, 1, -1)  # (B, 1, 1, T/sub_rate)
        dummy_embed_xs = torch.full((1, masks.size(2), 1), 0.0,
                                    device=xs.device)
        chunk_masks = add_optional_chunk_mask(dummy_embed_xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        chunk_masks = chunk_masks.unsqueeze(1).type_as(xs).expand(
            B, 1, mask_pad.size(3), mask_pad.size(3)
        )  # (B, 1, T/sub_rate, T/sub_rate)
        fake_cache = torch.zeros([0, 0, 0, 0],
                                 dtype=xs.dtype, device=xs.device)

        for i, layer in enumerate(self.encoders):
            xs, _, _ = layer(
                xs, chunk_masks, mask_pad, fake_cache, fake_cache)
        if self.final_norm == "layer_norm":
            xs = xs.squeeze(2).transpose(1, 2).contiguous()
            xs = self.after_norm(xs)  # (B, T, size)
        else:
            raise NotImplementedError()
        return xs, masks

    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Fake API, for passing JIT check.
        """
        return xs, xs, xs

    def forward_chunk_j3(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_mask: torch.Tensor = torch.ones((0, 0, 0, 0), dtype=torch.float),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features with 4-dimension data-flow.

        FIXME(xcsong): update doc.
        Args:
            xs (torch.Tensor): (#batch=1, c=1, time, mel-dim), j3 mode
                or (#batch=1, time, mel-dim), server mode
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): if offset < 0: j3 mode, never used.
                          else: server mode
            required_cache_size (int): if offset < 0: j3 mode, never used
                                       else: server mode, get next_cache_start
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, hidden-dim, 1, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        # 4D dataflow
        if offset >= 0:  # server mode
            xs = xs.unsqueeze(1)  # (B, T, F) -> (B, C=1, T, F)
        else:  # j3 mode
            xs = xs  # (B, C=1, T, F)

        # subsampling
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        # NOTE(xcsong): Before embed, shape(xs) is (b=1, c=1, time, mel-dim)
        xs, _ = self.embed(xs)
        # NOTE(xcsong): After  embed, shape(xs) is (b=1, h-dim, 1, chunk_size)
        b, size, _, chunk_size = xs.size()
        elayers = len(self.encoders)
        # NOTE(xcsong):
        #   att_cache in standard conformer: (elayer, head, cache_t1, d_k * 2)
        #   att_cache in j3 conformer: (1, elayer * head, cache_t1, d_k * 2)
        #   cnn_cache in standard conformer: (elayer, h-dim, 1, cache_t2)
        #   cnn_cache in j3 conformer: (1, elayer * h-dim, 1, cache_t2)
        #   Why ? Horizon compiler treats the 1st axis as batch-axis, `elayers`
        #   dose not matches the meaning of `batch` and may result in
        #   compilation errors.
        _, head_mul_elayers, cache_t, _ = att_cache.size()
        head = head_mul_elayers // elayers
        attention_key_size = cache_t + chunk_size
        # elayers * [1, head, cache_t1, d_k * 2]
        att_cache = torch.split(att_cache, head, dim=1)
        # elayers * [1, h-dim, 1, cache_t2]
        cnn_cache = torch.split(cnn_cache, size, dim=1)

        # next_cache_start
        if offset >= 0:  # server mode, dynamic slice
            if required_cache_size < 0:
                next_cache_start = 0
            elif required_cache_size == 0:
                next_cache_start = attention_key_size
            else:
                next_cache_start = max(
                    attention_key_size - required_cache_size, 0)
        else:  # j3 mode, static slice
            next_cache_start = chunk_size

        # encoder layers
        new_cnn_caches, new_att_caches = [], []
        for i, layer in enumerate(self.encoders):
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i]) is (1, hidden-dim, 1, cache_t2)
            xs, new_att_cache, new_cnn_cache = layer(
                xs, mask_attn=att_mask,
                att_cache=att_cache[i],
                cnn_cache=cnn_cache[i],
            )
            # NOTE(xcsong): After layer.forward
            #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (1, hidden-dim, 1, cache_t2)
            new_att_caches.append(new_att_cache[:, :, next_cache_start:, :])
            new_cnn_caches.append(new_cnn_cache)

        new_att_caches = torch.cat(new_att_caches, 1)
        new_cnn_caches = torch.cat(new_cnn_caches, 1)

        # final normalization
        if self.final_norm == "layer_norm":
            xs = xs.squeeze(2).transpose(1, 2).contiguous()
            xs = self.after_norm(xs)  # (B, T, size)
            # NOTE(xcsong): 4D in, 4D out
            xs = xs.transpose(1, 2).contiguous().unsqueeze(2)  # (B, C, 1, T)
        else:
            raise NotImplementedError()

        return xs, new_att_caches, new_cnn_caches

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder attention key/value cache
            2. convolution cache in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            xs (torch.Tensor): (#batch=1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        cnn_cache = torch.zeros(
            [1, self.output_size() * len(self.encoders), 0, 0],
            dtype=xs.dtype, device=xs.device)
        att_cache = torch.zeros(
            [1, self.encoders[0].attn.h * len(self.encoders), 0, 0],
            dtype=xs.dtype, device=xs.device)
        outputs = []
        offset = 0  # server mode
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]  # (1, decoding_window, mel)
            print(chunk_xs.size(), cnn_cache.size(), att_cache.size())
            (y, att_cache, cnn_cache) = self.forward_chunk_j3(
                chunk_xs, offset, required_cache_size, att_cache, cnn_cache)
            y = y.squeeze(2).transpose(1, 2).contiguous()
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)  # (1, time, size)
        masks = torch.ones(1, ys.size(1), device=ys.device, dtype=torch.bool)
        masks = masks.unsqueeze(1)
        return ys, masks
