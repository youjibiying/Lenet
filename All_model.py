# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:41:35 2019

@author: youjibiying
"""
import torch
import torch.nn as nn
from basic_block import ConvBlock, sequential, activation
import torch.nn as nn
import torch.nn.functional as F
import math


# import torch.utils.data as Data
# import matplotlib.pyplot as plt


class BCN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, num_features=4, model_num=2, act_type='prelu', norm_type=None):
        super(BCN, self).__init__()

        if model_num == 2:
            stride = 1
            padding = 2
            kernel_size = 3
        elif model_num == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif model_num == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif model_num == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        self.num_features = num_features
        self.model_num = model_num

        # Conv(3; 4m) and Conv(1; m). m denotes the base

        self.conv_in = ConvBlock(in_channels, 4 * num_features,
                                 kernel_size=3, stride=stride,
                                 act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4 * num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)
        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3,
                                  act_type=None, norm_type=norm_type)
        self.fc1 = nn.Linear(10 * 28 * 28, 500)
        self.fc2 = nn.Linear(500, 10)

        # basic block
        # self.block = FeedbackBlock(num_features, num_groups, model_num, act_type, norm_type)

        # reconstruction block
        # uncomment for pytorch 0.4.0
        # self.upsample = nn.Upsample(scale_factor=model_num, mode='bilinear')

        # self.out = DeconvBlock(num_features, num_features,
        #                        kernel_size=kernel_size, stride=stride, padding=padding,
        #                        act_type='prelu', norm_type=norm_type)
        # self.conv_out = ConvBlock(num_features, out_channels,
        #                           kernel_size=3,
        #                           act_type=None, norm_type=norm_type)

        # self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.feat_in(x)
        x = self.conv_out(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x  # return x for visualization


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)


def Lenet_activate():
    model = nn.Sequential(
        # Lambda(preprocess),
        nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(24, 32, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        # nn.Conv2d(16, 10, kernel_size=3, stride=1, padding=0),
        # nn.ReLU(),
        nn.MaxPool2d(2),
        Lambda(lambda x: x.view(x.size(0), -1)),
        nn.Linear(in_features=512, out_features=120, bias=True),
        nn.Linear(in_features=120, out_features=84, bias=True),
        nn.Linear(in_features=84, out_features=10, bias=True),
        nn.LogSoftmax(dim=1)
    )
    return model


class Lenet(nn.Module):

    def __init__(self, model_num=1, in_channels=1, out_channels=10, num_features=1, num_features1=1, drop_rate=0.5,
                 act_type='lrelu',
                 norm_type=None):
        super(Lenet, self).__init__()
        if model_num == 0:
            stride = 1
            kernel_size = 3
            kernel_size2 = 3
        if model_num == 1:
            stride = 1
            kernel_size = 5
            kernel_size2 = 5
        elif model_num == 2:
            stride = 1
            kernel_size = 7
            kernel_size2 = 7
        elif model_num == 3:
            stride = 2
            kernel_size = 3
            kernel_size2 = 3
        elif model_num == 4:
            stride = 1
            kernel_size = 5
            kernel_size2 = 3


        self.kernel_size = kernel_size
        self.kernel_size2 = kernel_size2
        self.cov1nums1 = (28 - kernel_size) // stride + 1  # 第一次卷积完成之后的图像大小
        self.cov1nums2 = (self.cov1nums1 // 2 - kernel_size2) // stride + 1  # 第二次卷积完成之后的图像大小
        self.fc1nums2 = ((self.cov1nums2 // 2) ** 2) * 16 * num_features1

        # self.conv_in = ConvBlock(1, 6 * num_features,
        #                          kernel_size=3,
        #                          act_type=act_type, norm_type=norm_type)
        # self.feat_in = ConvBlock(1,
        #                          kernel_size=1,
        #                          act_type=act_type, norm_type=norm_type)
        self.conv1 = nn.Conv2d(1, 6 * num_features, kernel_size=kernel_size2, stride=stride)
        self.conv2 = nn.Conv2d(6 * num_features, 16 * num_features1, kernel_size=kernel_size2, stride=stride)
        self.Dropout_rate = drop_rate
        self.fc1 = nn.Linear(self.fc1nums2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.act = activation(act_type) if act_type else None
        # self.in_out = sequential(self.conv1, self.act, self.conv2, self.act, self.fc1, self.act, self.fc2, self.act,
        #                          self.fc3)
        # self.reset_parameters()   # 参数初始化

    def reset_parameters(self):
        n = 1
        for k in [self.kernel_size, self.kernel_size]:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        for weight in self.parameters():
            if len(weight.size()) > 1:
                
                # torch.nn.init.kaiming_uniform_(weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.kaiming_normal_(weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            # nn.init.constant_(weight, 1)
            # weight.data.normal_(mean=0, std=stdv) #用分布将每一组参数都进行初始化，其中1*6*5*5 为第一层，6*1为阈值；16*6*5*5 。。。
            # weight.data.uniform_(-stdv,stdv)

    def forward(self, x):
        # 2x2 Max pooling
        # x = self.conv_in(x)
        # x = self.feat_in(x)
        x = F.max_pool2d(self.act(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(self.act(self.conv2(x)), 2)
        # x = F.dropout(x, p=self.Dropout_rate, training=self.training)
        x = x.view(-1, self.num_flat_features(x))
        x = self.act(self.fc1(x))
        # x = F.dropout(x, p=self.Dropout_rate, training=self.training)
        x = self.act(self.fc2(x))
        # x = F.dropout(x, p=self.Dropout_rate, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # LSTM 效果要比 nn.RNN() 好多了
            input_size=28,  # 图片每行的数据像素点
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)  # 输出层

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        return F.log_softmax(out, dim=1)
