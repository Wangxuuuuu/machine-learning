from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # As mentioned in the assignment, blocks of the downsampling path should have the
        # following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
        layers_conv_down = []
        layers_bn_down = []
        layers_pooling = []

        current_in_channels = input_size
        for i in range(self.num_down_layers):
            # 1. 卷积层: 提取特征，通道数从 current_in_channels 变为 down_filter_sizes[i]
            layers_conv_down.append(nn.Conv2d(current_in_channels, down_filter_sizes[i], kernel_size=kernel_sizes[i], padding=conv_paddings[i]))
            # 2. 批归一化层: 加速收敛，防止过拟合
            layers_bn_down.append(nn.BatchNorm2d(down_filter_sizes[i]))
            # 3. 最大池化层: 降低特征图尺寸 (下采样)。
            # 关键点: return_indices=True，必须保存最大值的索引位置，用于后续的上采样恢复
            layers_pooling.append(nn.MaxPool2d(kernel_size=pooling_kernel_sizes[i], stride=pooling_strides[i], return_indices=True))
            current_in_channels = down_filter_sizes[i]

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # As mentioned in the assignment, blocks of the upsampling path should have the
        # following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        layers_conv_up = []
        layers_bn_up = []
        layers_unpooling = []

        current_in_channels = down_filter_sizes[-1]
        for i in range(self.num_up_layers):
            # 1. 最大反池化层: 恢复特征图尺寸 (上采样)。
            # 需要在 forward 中传入对应的 indices 才能精确恢复形状
            layers_unpooling.append(nn.MaxUnpool2d(kernel_size=pooling_kernel_sizes[i], stride=pooling_strides[i]))
            # 2. 卷积层: 调整通道数，逐步恢复特征细节
            layers_conv_up.append(nn.Conv2d(current_in_channels, up_filter_sizes[i], kernel_size=kernel_sizes[i], padding=conv_paddings[i]))
            # 3. 批归一化层
            layers_bn_up.append(nn.BatchNorm2d(up_filter_sizes[i]))
            current_in_channels = up_filter_sizes[i]

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)

        self.relu = nn.ReLU(True)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)
        # 最终分类层: 使用 1x1 卷积将通道数映射为类别数 (11类: 背景 + 0-9数字)
        self.conv_final = nn.Conv2d(up_filter_sizes[-1], 11, kernel_size=1)

    def forward(self, x):
        # 用于存储下采样过程中产生的最大池化索引 (indices)
        indices_list = []
        
        # Downsampling (编码器路径)
        for i in range(self.num_down_layers):
            x = self.layers_conv_down[i](x)
            x = self.layers_bn_down[i](x)
            x = self.relu(x)
            # 执行最大池化，并获取 indices (最大值位置)
            x, indices = self.layers_pooling[i](x)
            indices_list.append(indices)
            
        # Upsampling (解码器路径)
        for i in range(self.num_up_layers):
            # 执行最大反池化
            # 注意: 使用 indices_list[-(i+1)] 获取对应的下采样层的索引 (倒序使用)
            x = self.layers_unpooling[i](x, indices_list[-(i+1)])
            x = self.layers_conv_up[i](x)
            x = self.layers_bn_up[i](x)
            x = self.relu(x)
            
        # 输出层: 生成最终的分割图 logits
        x = self.conv_final(x)
        return x


def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
