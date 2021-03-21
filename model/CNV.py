# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn
from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d

from brevitas.nn import QuantConv2d, QuantLinear
from utils.common import make_quant_conv2d, make_quant_linear, make_quant_relu


CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 3


class CNV(Module):

    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch):
        super(CNV, self).__init__()

        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(
            ConvBlock(
                in_channels=in_ch,
                out_channels=CNV_OUT_CH_POOL[0][0],
                kernel_size=KERNEL_SIZE,
                weight_bit_width=in_bit_width,
                act_bit_width=act_bit_width
                )
        )

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL[1:]:
            self.conv_features.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=KERNEL_SIZE,
                    weight_bit_width=weight_bit_width,
                    act_bit_width=act_bit_width
                    )
            )
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(
                make_quant_linear(
                    in_channels=in_features,
                    out_channels=out_features,
                    bias=False,
                    bit_width=weight_bit_width
                )
            )
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(
                make_quant_relu(
                    bit_width=act_bit_width
                )
            )

        self.linear_features.append(
            make_quant_linear(
                in_channels=LAST_FC_IN_FEATURES,
                out_channels=num_classes,
                bias=False,
                bit_width=weight_bit_width
            )
        )

    def forward(self, x):
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x

class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 weight_bit_width,
                 act_bit_width,
                 stride=1,
                 padding=0,
                 groups=1,
                 bn_eps=1e-4,
                 activation_scaling_per_channel=False):
        super(ConvBlock, self).__init__()
        self.conv = make_quant_conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      groups=groups,
                                      bias=False,
                                      bit_width=weight_bit_width)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activation = make_quant_relu(bit_width=act_bit_width,
                                          per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                                          scaling_per_channel=activation_scaling_per_channel,
                                          return_quant_tensor=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


def cnv(weight_bit_width, act_bit_width, in_bit_width):
    num_classes = 10
    in_channels = 3
    net = CNV(weight_bit_width=weight_bit_width,
              act_bit_width=act_bit_width,
              in_bit_width=in_bit_width,
              num_classes=num_classes,
              in_ch=in_channels)
    return net

cnv(8,8,8)