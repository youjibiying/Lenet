# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:33:43 2019

@author: youjibiying
"""

import torch
from torch import nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
from All_model import RNN
torch.manual_seed(1)  # reproducible

# Hyper Parameters
EPOCH = 1  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 64
TIME_STEP = 28  # rnn 时间步数 / 图片高度
INPUT_SIZE = 28  # rnn 每步输入值 / 图片每行像素
LR = 0.01  # learning rate
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 Fasle

# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist/',  # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,  # 没下载就下载, 下载了就不用再下了
)

# plot one example
print(train_data.train_data.size())  # (60000, 28, 28)
print(train_data.train_labels.size())  # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# Data Loader for easy mini-batch return in training# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
# 为了节约时间, 我们测试时只测试前2000个
test_x = test_data.test_data.type(torch.FloatTensor)[:2000] / 255.  # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy()[:2000]  # covert to numpy array


def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.data.fill_(1.0)
#        print(m.weight)
    elif type(m) == nn.Conv2d:
        m.weight.data.normal_(0, math.sqrt(2. / 5))


#    elif typr(m)==



rnn = RNN()
rnn.apply(init_weights)
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

accur = []
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
        b_x = b_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 5 == 0:
            test_output = rnn(test_x)  # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), \
                  '| test accuracy: %.2f' % accuracy)  # print 10 predictions from test data
            accur.append(accuracy)

plt.plot([i for i in range(0, 5 * len(accur), 5)], accur)
plt.show()




test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')


torch.save(rnn.state_dict(), './model/rnn_net_params.pkl')  # save only the parameters


from All_model import RNN
import torch
def restore_params(m):  # 恢复模型以及测试结果
    # restore only the parameters in rnn1
    rnn1 = RNN()

    # copy net1's parameters into rnn1
    rnn1.load_state_dict(torch.load('./model/rnn_net_params.pkl'))
    test_output = rnn1(test_x[:m].view(-1, 28, 28))
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    accuracy = sum(pred_y == test_y[:m]) / m
    print('test accuracy:%.2f' % accuracy)

