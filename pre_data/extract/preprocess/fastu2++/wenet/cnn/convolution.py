#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c)  2021 Di Wu (di.wu@mobvoi.com)
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
"""ConvolutionModule definition. Conv2d version."""

from typing import Tuple

import torch
from torch import nn
from typeguard import check_argument_types


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model. Conv2d version."""
    def __init__(self,
                 in_channels: int,
                 inner_channels: int,
                 kernel_size: int = 7,
                 activation: nn.Module = nn.ReLU(),
                 causal: bool = False,
                 bias: bool = True):
        """Construct an ConvolutionModule object.

        Args:
            in_channels (int): The number of input channels.
            inner_channels (int): Inner channels of depthwise_conv.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not

        """
        assert check_argument_types()
        super().__init__()

        self.pointwise_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inner_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.BatchNorm2d(inner_channels),
            activation,
        )

        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
            self.pad = nn.ConstantPad2d(
                padding=(self.lorder, 0, 0, 0),
                value=0.0,
            )
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
            self.pad = nn.Identity()

        # TODO(xcsong): Remove activation in dw_conv may be beneficial,
        #               ref: Xception, section4.7.
        # NOTE(xcsong): Different from wenet.transformer.convolution, we use
        #               ReLu instead of GLU to activate the output of pw_conv1,
        #               thus the input-channels of dw_conv should be
        #               2*in_channels (aka inner_channels).
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                inner_channels,
                inner_channels,
                (1, kernel_size),
                stride=1,
                padding=(0, padding),
                groups=inner_channels,
                bias=bias,
            ),
            nn.BatchNorm2d(inner_channels),
            activation,
        )

        self.pointwise_conv2 = nn.Sequential(
            nn.Conv2d(
                inner_channels,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.BatchNorm2d(in_channels),
        )
        self.pw_quant1 = torch.quantization.QuantStub()
        self.pw_quant2 = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def fuse_modules(self):
        torch.quantization.fuse_modules(
            self,
            [['pointwise_conv1.0', 'pointwise_conv1.1', 'pointwise_conv1.2']],
            inplace=True)
        torch.quantization.fuse_modules(
            self,
            [['depthwise_conv.0', 'depthwise_conv.1', 'depthwise_conv.2']],
            inplace=True)
        torch.quantization.fuse_modules(
            self,
            [['pointwise_conv2.0', 'pointwise_conv2.1']],
            inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor,
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, in_channels, 1, time).
            mask_pad (torch.Tensor): used for batch padding
                (#batch, 1, 1, time)
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, in_channels, 1, cachetime)

        Returns:
            torch.Tensor: Output tensor (#batch, in_channels, 1, time).
            torch.Tensor: Cache tensor (#batch, in_channels, 1, cachetime).

        """
        if mask_pad.size(0) > 0:
            x = x * mask_pad

        if self.lorder > 0:
            if cache.size(3) == 0:  # cachetime == 0
                x = self.pad(x)
            else:
                assert cache.size(0) == x.size(0)  # batch
                assert cache.size(1) == x.size(1)  # channel
                x = torch.cat((cache, x), dim=3)   # (b, c, 1, cache_t + t)
            assert (x.size(3) > self.lorder)
            new_cache = x[:, :, :, -self.lorder:]  # (b, c, 1, cache_t)
        else:
            # It's better we just return None if no cache is requried,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros([0, 0, 0, 0],
                                    dtype=x.dtype, device=x.device)

        x = self.pw_quant1(x)
        x = self.pointwise_conv1(x)  # (batch, inner_channel, 1, cache_t + t)
        # NOTE(xcsong): it is better to run depthwise_conv in float32 mode. see
        #      https://discuss.pytorch.org/t/got-slow-speed-on-quantized-
        #       model-with-fbgemm-on-x86/74439
        x = self.dequant(x)
        x = self.depthwise_conv(x)   # (batch, inner_channel, 1, t)
        x = self.pw_quant2(x)
        x = self.pointwise_conv2(x)  # (batch, in_channel,    1, t)
        x = self.dequant(x)

        if mask_pad.size(0) > 0:
            x = x * mask_pad

        return x, new_cache
