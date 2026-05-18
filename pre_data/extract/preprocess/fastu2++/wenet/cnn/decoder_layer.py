#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Shigeki Karita
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
"""Decoder self-attention layer definition."""
import torch
from torch import nn


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Inter-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
    """
    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def fuse_modules(self):
        self.self_attn.fuse_modules()
        self.src_attn.fuse_modules()
        self.feed_forward.fuse_modules()

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> torch.Tensor:
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, size, 1, time_out).
            tgt_mask (torch.Tensor): Mask for input tensor
                (#batch, 1, time_out, time_out).
            memory (torch.Tensor): Encoded memory
                (#batch, size, 1, time_in).
            memory_mask (torch.Tensor): Encoded memory mask
                (#batch, 1, 1, time_in).
            cache (torch.Tensor): cached tensors.
                (#batch, time_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, size, 1, t_out).

        """
        residual = tgt

        if cache.size(0) == 0:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0], self.size, 1, tgt.shape[3] - 1,
            ), "{cache.shape} == {(tgt.shape[0], self.size, 1, tgt.shape[1] - 1)}"
            tgt_q = tgt[:, :, :, -1:]
            residual = residual[:, :, :, -1:]
            tgt_q_mask = tgt_mask[:, :, -1:, :]

        x = residual + self.dropout(
            self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0])

        residual = x
        x = residual + self.dropout(
            self.src_attn(x, memory, memory, memory_mask)[0])

        residual = x
        x = residual + self.dropout(self.feed_forward(x))

        if cache.size(0) > 0:
            x = torch.cat([cache, x], dim=3)

        return x
