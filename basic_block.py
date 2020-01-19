# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:37:10 2019

@author: youjibiying
"""

# import torch
# import numpy as np
# from sklearn.datasets import make_blobs,make_circles,make_moons
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

import torch.nn as nn
from collections import OrderedDict
import sys


def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)  # 是否加入空洞，dilation==1则加入边缘
    padding = (kernel_size - 1) // 2
    return padding


def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    elif act_type == 'sigmoid':
        layer = nn.Sigmoid()
    elif act_type == 'tanh':
        layer = nn.Tanh()
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!' % act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] Normalization layer [%s] is not implemented!' % norm_type)
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None

    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('[ERROR] Padding layer [%s] is not implemented!' % pad_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict' % sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)

        elif isinstance(module, nn.Module):
            modules.append(module)
    # print(modules)
    return nn.Sequential(*modules)


def ConvBlock(in_channels, out_channels, kernel_size, drop_rate=0, stride=1, dilation=1, bias=True,
              valid_padding=True, padding=0, \
              act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!' % sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias)
    drop = nn.Dropout(drop_rate)
    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, drop, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)
#
# out_channels=3
# norm_type='bn'
# act_type='relu'
# pad_type='zero'
# p = None
# conv = nn.Conv2d(3, 3, 3, 2, padding=True, dilation=1, bias=True)
# act = activation(act_type) if act_type else None
# n = norm(out_channels, norm_type) if norm_type else None
# net=sequential(p, conv, n, act)
