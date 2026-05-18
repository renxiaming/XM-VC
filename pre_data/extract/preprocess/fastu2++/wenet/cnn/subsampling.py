#!/usr/bin/env python3
# Copyright (c) 2019 Di Wu (di.wu@mobvoi.com)
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
"""Subsampling layer definition. Fully-convolutional version."""

import torch

from typing import Tuple

from wenet.transformer.subsampling import BaseSubsampling


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, idim: int, odim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Conv2d(odim * (((idim - 1) // 2 - 1) // 2),
                                   odim, 1, 1)
        self.dropout = torch.nn.Dropout(dropout_rate, inplace=True)
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def fuse_modules(self):
        torch.quantization.fuse_modules(
            self, [['conv.0', 'conv.1']], inplace=True)
        torch.quantization.fuse_modules(
            self, [['conv.2', 'conv.3']], inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, 1, time, idim).
            masks (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, odim, 1, time'),
                where time' = time // 4.

        """
        x = self.quant(x)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = x.transpose(2, 3).contiguous().view(b, c * f, 1, t)
        x = self.dropout(self.out(x))
        if masks.size(0) > 0:
            masks = masks[:, :, :-2:2][:, :, :-2:2]
        x = self.dequant(x)
        return x, masks


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, idim: int, odim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Conv2d(
            odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2),
            odim, 1, 1)
        self.dropout = torch.nn.Dropout(dropout_rate, inplace=True)
        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, 1, time, idim).
            masks (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            torch.Tensor: positional encoding
        """
        x = self.quant(x)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = x.transpose(2, 3).contiguous().view(b, c * f, 1, t)
        x = self.dropout(self.out(x))
        if masks.size(0) > 0:
            masks = masks[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]
        x = self.dequant(x)
        return x, masks
