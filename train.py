# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:26:30 2019

@author: youjibiying
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import math
from All_model import BCN, CNN, Lenet
import pickle
import os


#  Softmax+Log+NLLLoss==交叉熵


def save_variable_Loss(v, filename):
    filename = './Loss/' + filename + '_loss.txt'
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def save_variable_accur(v, filename):
    filename = './Loss/' + filename + '_accur.txt'
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def train(args, model, device, train_loader, optimizer, epoch, lossRecord):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  # Softmax+Log+NLLLoss==交叉熵
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:  # 展示区间
            lossRecord.append(loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, accur):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss ,
            # reduction default is "mean"https://pytorch.org/docs/stable/nn.functional.html#nll-loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accur.append(correct / len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='writing digital recognition')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} # num_workers 是多线程的意思
    # pin_memory（bool, 可选）– 如果设置为True，数据加载器会在返回前将张量拷贝到CUDA锁页内存。
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=True, download=not os.path.exists('../data/mnist'),
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # model1 = BCN().to(device)

    model = Lenet(num_features=4, num_features1=2, act_type='lrelu').to(
        device)  # act_type=[sigmoid,tanh,Relu,prelu,lrelu]

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0001)
    print(model)
    accur = []
    lossRecord = []
    name = 'Lenet1_nums_features4_2_lrelu'  # 'Lenet_nums_features4_2_lrelu' #
    print("Start train! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, lossRecord)
        test(args, model, device, test_loader, accur)
    # save_variable_accur(accur, name)
    # save_variable_Loss(lossRecord, name)
    # if (args.save_model):
    #     torch.save(model.state_dict(), './model/' + name + '_params.pkl')  # save only the parameters
    #     print(model)
    #     print(">>>>>>>>>>>" + name + " Model has been Saved successfully! ")


if __name__ == '__main__':
    main()
