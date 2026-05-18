#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c)  2019 Shigeki Karita
#                2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
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
"""Attention layer definition, fully-convolutional version."""

import math
from typing import Tuple

import torch
from torch import nn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    eq: score(x_i, x_j) = e^(THETA(x_i)^T dot PHI(x_j))
        where THETA is linear_q and PHI is linear_k

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1),
            nn.BatchNorm2d(n_feat),
        )
        self.linear_k = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1),
            nn.BatchNorm2d(n_feat),
        )
        self.linear_v = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1),
            nn.BatchNorm2d(n_feat),
        )
        self.linear_out = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1),
            nn.BatchNorm2d(n_feat),
        )
        self.activation = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        # NOTE(xcsong): To allow 4D dataflow, we manually construct a
        #               4D tensor with shape [1, n_head, 1, 1],
        #               see https://horizonrobotics.feishu.cn/wiki/wikcnqUpjuJEKwex0t9SPBxOD7f?sheet=Zb0Or1&range=STQ
        self.register_buffer(
            "denom", torch.full((1, self.h, 1, 1), 1.0 / math.sqrt(self.d_k)))
        # NOTE(xcsong): QuantStub is just a place holder for quantize op,
        #   it needs to be unique since it has state. DeQuantStub is a
        #   place holder for dequantize op, but it does not need to be unique
        #   since it’s stateless.
        self.q_quant = torch.quantization.QuantStub()
        self.k_quant = torch.quantization.QuantStub()
        self.v_quant = torch.quantization.QuantStub()
        self.x_quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def fuse_modules(self):
        torch.quantization.fuse_modules(
            self, [['linear_q.0', 'linear_q.1']], inplace=True)
        torch.quantization.fuse_modules(
            self, [['linear_k.0', 'linear_k.1']], inplace=True)
        torch.quantization.fuse_modules(
            self, [['linear_v.0', 'linear_v.1']], inplace=True)
        torch.quantization.fuse_modules(
            self, [['linear_out.0', 'linear_out.1']], inplace=True)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        NOTE(xcsong): Input shape [N, C, H, W] is required by torch.Conv2d,
            the reason for choosing `W` as time-axis is to avoid unnecessary
            padding, see `width alignment` for more details:
            https://horizonrobotics.feishu.cn/docs/doccncwfEKITYRcQVA4Bx3nfPQf#9jhvRq

        Args:
            query (torch.Tensor): Query tensor (#batch, n_feat, 1, time1).
            key (torch.Tensor):   Key tensor   (#batch, n_feat, 1, time2).
            value (torch.Tensor): Value tensor (#batch, n_feat, 1, time2).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, head, time2, d_k).

        """
        n_batch, n_feat, _, time1 = query.size()
        _, _, _, time2 = key.size()
        q = self.linear_q(self.q_quant(query)).view(
            n_batch, self.h, self.d_k, time1)
        k = self.linear_k(self.k_quant(key)).view(
            n_batch, self.h, self.d_k, time2)
        v = self.linear_v(self.v_quant(value)).view(
            n_batch, self.h, self.d_k, time2)
        q = q.transpose(2, 3)  # (batch, head, time1, d_k)
        k = k.transpose(2, 3)  # (batch, head, time2, d_k)
        v = v.transpose(2, 3)  # (batch, head, time2, d_k)

        return self.dequant(q), self.dequant(k), self.dequant(v)

    def caculate_scores(self, query: torch.Tensor,
                        key: torch.Tensor) -> torch.Tensor:
        """Compute attention scores.

        Args:
            query (torch.Tensor): Query tensor (#batch, n_head, time1, d_k).
            key (torch.Tensor):   Key tensor   (#batch, n_head, time2, d_k).

        Returns:
            torch.Tensor: Score tensor (#batch, n_head, time1, time2).

        """
        key = key.transpose(2, 3)  # (batch, head, d_k, time2)
        scores = torch.matmul(query, key) * \
            self.denom  # (#b, n_head, time1, time2)
        return scores

    def forward_attention(self, value: torch.Tensor, scores: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, n_feat, 1, time1)
                weighted by the attention score (#batch, n_head, time1, time2).

        """
        n_batch, _, time1, _ = scores.size()
        if mask.size(0) > 0:  # training mode
            mask = mask.eq(0)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = self.activation(scores).masked_fill(mask, 0.0)
        else:  # eval mode
            attn = self.activation(scores)  # (batch, n_head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)

        x = x.transpose(2, 3).contiguous().view(
            n_batch, self.d_k * self.h, 1, time1
        )

        # (batch, n_feat, 1, time1)
        return self.dequant(self.linear_out(self.x_quant(x)))

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor,
                cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, n_feat, 1, time1).
            key (torch.Tensor):   Key tensor   (#batch, n_feat, 1, time2).
            value (torch.Tensor): Value tensor (#batch, n_feat, 1, time2).
            mask (torch.Tensor):  Mask tensor  (#batch, 1, time1, time2)
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2).

        Returns:
            torch.Tensor: Output tensor (#batch, n_feat, 1, time1).
            torch.Tensor: Cache tensor (1, head, cache_t + time2, d_k * 2).

        """
        # Step-1: forward qkv
        q, k, v = self.forward_qkv(query, key, value)

        # NOTE(xcsong): cache with shape [0, 0, 0, 0] means fake cache
        if cache.size(2) > 0:  # cache_t > 0
            k_cache, v_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat((k_cache, k), dim=2)
            v = torch.cat((v_cache, v), dim=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #               non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        # Step-2: calculate scores, (#batch, n_head, time1, time2)
        scores = self.caculate_scores(q, k)

        # Step-3: forward attention, (#batch, n_feat, 1, time1)
        return self.forward_attention(v, scores, mask), new_cache
