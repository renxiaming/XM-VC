#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Di Wu (di.wu@mobvoi.com)
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
"""Decoder definition. Conv2d Version."""
from typing import Tuple, List, Optional

import torch
from typeguard import check_argument_types

from wenet.cnn.attention import MultiHeadedAttention
from wenet.cnn.decoder_layer import DecoderLayer
from wenet.cnn.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.embedding import PositionalEncoding
from wenet.utils.common import get_activation
from wenet.utils.mask import (subsequent_mask, make_pad_mask)


class TransformerCnnDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        activation_type: str = "relu",
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"found: {input_layer}, only 'embed' is supported")

        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-12)
        self.output_layer = torch.nn.Conv2d(attention_dim, vocab_size, 1, 1, 0)
        self.num_blocks = num_blocks
        activation = get_activation(activation_type)
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     self_attention_dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate, activation),
                dropout_rate,
            ) for _ in range(self.num_blocks)
        ])

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def fuse_modules(self):
        for layer in self.decoders:
            layer.fuse_modules()

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: Optional[torch.Tensor] = None,
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, time, feat)
            memory_mask: encoder memory mask, (batch, 1, time)
            ys_in_pad: padded input token ids, int64 (batch, length)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, length,
                    vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with bidirectional decoder
                olens: (batch, )
        """
        # attention mask
        B, L = ys_in_pad.size()
        tgt_mask = ~make_pad_mask(ys_in_lens, L).unsqueeze(1)
        tgt_mask = tgt_mask.to(ys_in_pad.device)  # (B, 1, L)
        m = subsequent_mask(tgt_mask.size(-1),
                            device=tgt_mask.device).unsqueeze(0)  # (1, L, L)
        tgt_mask = tgt_mask & m  # (B, L, L)
        olens = tgt_mask.sum(1)

        # positional encoding
        x, _ = self.embed(ys_in_pad)  # (B, L, D)

        # decoder layers, 4D dataflow
        x = x.transpose(1, 2).contiguous().unsqueeze(2)  # (B, D, 1, L)
        memory = memory.transpose(
            1, 2).contiguous().unsqueeze(2)  # (B, D, 1, T)
        tgt_mask = tgt_mask.unsqueeze(1).type_as(x).expand(
            B, 1, L, L)  # int -> float, (B, 1, L, L)
        memory_mask = memory_mask.unsqueeze(1).type_as(x).expand(
            B, 1, 1, -1)  # int -> float, (B, 1, 1, T)
        for layer in self.decoders:
            x = layer(x, tgt_mask, memory, memory_mask)

        # final norm and projection
        x = self.quant(x)
        x = x.squeeze(2).transpose(1, 2).contiguous()  # (B, L, D)
        x = self.after_norm(x)
        x = x.transpose(1, 2).contiguous().unsqueeze(2)  # (B, D, 1, L)
        x = self.output_layer(x)  # (B, vocab_size, 1, L)
        x = x.squeeze(2).transpose(1, 2).contiguous()  # (B, L, vocab_size)
        x = self.dequant(x)
        return x, torch.tensor(0.0), olens

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, time, feat)
            memory_mask: encoded memory mask, (batch, 1, time)
            tgt: input token ids, int64 (batch, length)
            tgt_mask: input token mask,  (batch, length)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, length-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        # positional encoding
        x, _ = self.embed(tgt)

        # decoder layers, 4D dataflow
        x = x.transpose(1, 2).contiguous().unsqueeze(2)  # (B, D, 1, L)
        memory = memory.transpose(
            1, 2).contiguous().unsqueeze(2)  # (B, D, 1, T)
        tgt_mask = tgt_mask.unsqueeze(1).type_as(x).expand(
            B, 1, L, L)  # int -> float, (B, 1, L, L)
        memory_mask = memory_mask.unsqueeze(1).type_as(x).expand(
            B, 1, 1, -1)  # int -> float, (B, 1, 1, T)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = torch.zeros((0, 0, 0, 0))
            else:
                c = cache[i]
            x = decoder(x, tgt_mask, memory, memory_mask, cache=c)
            new_cache.append(x)

        # final norm and projection
        x = x.squeeze(2).transpose(1, 2).contiguous()
        y = self.after_norm(x)
        y = y.transpose(1, 2).contiguous().unsqueeze(2)  # (B, D, 1, L)
        y = torch.log_softmax(self.output_layer(y[:, :, :, -1:]), dim=1)
        y = y.squeeze(2).transpose(1, 2).contiguous().squeeze(1)  # (B, v_size)
        return y, new_cache


class BiTransformerCnnDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        r_num_blocks: the number of right to left decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        r_num_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        activation_type: str = "relu",
    ):

        assert check_argument_types()
        super().__init__()
        self.left_decoder = TransformerCnnDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, activation_type)

        self.right_decoder = TransformerCnnDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            r_num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, activation_type)

    def fuse_modules(self):
        self.left_decoder.fuse_modules()
        self.right_decoder.fuse_modules()

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor,
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, time, feat)
            memory_mask: encoder memory mask, (batch, 1, time)
            ys_in_pad: padded input token ids, int64 (batch, length)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: padded input token ids, int64 (batch, length),
                used for right to left decoder
            reverse_weight: used for right to left decoder
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, length,
                    vocab_size) if use_output_layer is True,
                r_x: x: decoded token score (right to left decoder)
                    before softmax (batch, length, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        l_x, _, olens = self.left_decoder(memory, memory_mask, ys_in_pad,
                                          ys_in_lens)
        r_x = torch.tensor(0.0)
        if reverse_weight > 0.0:
            r_x, _, olens = self.right_decoder(memory, memory_mask, r_ys_in_pad,
                                               ys_in_lens)
        return l_x, r_x, olens

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, time, feat)
            memory_mask: encoded memory mask, (batch, 1, time)
            tgt: input token ids, int64 (batch, length)
            tgt_mask: input token mask,  (batch, length)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, length-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        return self.left_decoder.forward_one_step(memory, memory_mask, tgt,
                                                  tgt_mask, cache)
