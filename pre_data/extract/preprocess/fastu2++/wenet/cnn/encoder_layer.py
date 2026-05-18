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
"""Encoder layer/block definition."""

from typing import Tuple

import torch
from torch import nn


class ConformerCnnEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        attn (torch.nn.Module): attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance. `PositionwiseFeedForward` instance can
             be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
    """
    def __init__(
        self,
        size: int,
        attn: nn.Module,
        feed_forward: nn.Module,
        feed_forward_macaron: nn.Module,
        conv_module: nn.Module,
        dropout_rate: float = 0.1,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module

        # NOTE(xcsong): To allow 4D dataflow, we manually construct a
        #               4D tensor with shape [1, size, 1, 1],
        #               see https://horizonrobotics.feishu.cn/wiki/wikcnqUpjuJEKwex0t9SPBxOD7f?sheet=Zb0Or1&range=STQ
        self.register_buffer("ff_scale", torch.full((1, size, 1, 1), 0.5))
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def fuse_modules(self):
        self.attn.fuse_modules()
        self.feed_forward.fuse_modules()
        self.feed_forward_macaron.fuse_modules()
        self.conv_module.fuse_modules()

    def forward(
        self,
        x: torch.Tensor,
        mask_attn: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        mask_pad: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, size, 1, time)
            mask_attn (torch.Tensor): Mask tensor for attn (#b, 1, time，time).
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，1, time)
            cnn_cache (torch.Tensor): Cache tensor for cnn_module,
                (1, size, 1, cache_time1)
            att_cache (torch.Tensor): Cache tensor for KEY & VALUE,
                (1, head, cache_time2, d_k * 2), head * d_k == size.

        Returns:
            torch.Tensor: Output tensor (#batch, size, 1, time).
            torch.Tensor: Cnn cache tensor (1, size, 1, cache_t1).
            torch.Tensor: Att cache tensor (1, head, cache_t2 + time, d_k * 2).

        """
        # macaron feed forward module.
        residual = x  # (#b, size, 1, chunk_size)
        x = residual + self.ff_scale * \
            self.dropout(self.feed_forward_macaron(x))

        # multi-headed self-attention module.
        residual = x  # (#b, size, 1, chunk_size)
        x_att, new_att_cache = self.attn(x, x, x, mask_attn, att_cache)
        x = residual + self.dropout(x_att)

        # convolution module.
        residual = x  # (#b, size, 1, chunk_size)
        x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
        x = residual + self.dropout(x)

        # feed forward module.
        residual = x  # (#b, size, 1, chunk_size)
        x = residual + self.ff_scale * \
            self.dropout(self.feed_forward(x))

        return x, new_att_cache, new_cnn_cache
