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
"""Positionwise feed forward layer definition. fully-convolutional version"""

import torch


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer. fully-convolutional version.

    FeedForward are applied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function

    """
    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU()):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()

        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(idim, hidden_units,
                            kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(hidden_units),
            activation,
            torch.nn.Dropout(dropout_rate),
        )

        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_units, idim,
                            kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(idim),
        )

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def fuse_modules(self):
        torch.quantization.fuse_modules(
            self, [['conv_1.0', 'conv_1.1', 'conv_1.2']], inplace=True)
        torch.quantization.fuse_modules(
            self, [['conv_2.0', 'conv_2.1']], inplace=True)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, D, 1, T)

        Returns:
            output tensor, (B, D, 1, T)

        """
        return self.dequant(self.conv_2(self.conv_1(self.quant(xs))))
