# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:23:51 2019

@author: youjibiying
"""

import pickle
import matplotlib.pyplot as plt

loss = []
accur = []
log_interval = 50
plt.figure(0)
model_name = ['Lenet_nums_features4_2_lrelu']#'BCN', 'RNN']
model_name_regulation = ['Lenet', 'Lenet_regularization_0.01', 'Lenet_regularization_0.001' \
    , 'Lenet_regularization_0.0001']#
model_name_dropout = ['Lenet_regularization_0.0001', 'Lenet_dropout_1', 'Lenet_dropout_2', 'Lenet_dropout_3']

model_name_num_feature = ['Lenet_regularization_0.0001', 'Lenet_nums_features2', \
                          'Lenet_nums_features3', 'Lenet_nums_features4_2', 'Lenet_nums_features5_2']
model_name_kernel_size = ['Lenet_nums_features4_2_3', 'Lenet_nums_features4_2_5', \
                          'Lenet_nums_features4_2_7', 'Lenet_nums_features4_2_3_2', 'Lenet_nums_features4_2_5_3_1']
model_name_initial = ['Lenet_regularization_0.0001', 'Lenet_Uniform', 'Lenet_norm', 'Lenet_kaiming_uniform',
                      'Lenet_constant','Lenet_kaiming_normal']
#
model_name_activate = ['Lenet_nums_features4_2_relu','Lenet_nums_features4_2_lrelu','Lenet_nums_features4_2_prelu',
              'Lenet_nums_features4_2_sigmoid','Lenet_nums_features4_2_tanh']
#
model_name1 = ['Lenet_nums_features4_2_lrelu','Lenet1_nums_features4_2_lrelu','Lenet2_nums_features4_2_lrelu']
#_structure
for name in model_name:
    with open('E:/pythonCode/dataMining/project/CNN/Loss/' + name + '_loss.txt', 'rb') as f1:
        d1 = pickle.load(f1)
    with open('E:/pythonCode/dataMining/project/CNN/Loss/' + name + '_accur.txt', 'rb') as f2:
        d2 = pickle.load(f2)
    loss.append(d1)
    accur.append(d2)
    #    if name=='Lenet_norm':
    #        continue
    plt.plot([i for i in range(1, 1 + len(d1) * log_interval, log_interval)], d1, label=name)
# lenet
plt.legend()
plt.xlabel("Times")
plt.title("Loss of Model")
plt.show()
plt.figure(1)
for i in range(len(accur)):
    plt.plot([i + 1 for i in range(len(d2))], accur[i][:], label=model_name[i])
plt.legend()
plt.xlabel("Epochs")
plt.title("Accuracy of Model")
plt.show()


def plot(*arg):
    L = []
    for k in arg:
        L.append(k)
    plt.figure(2)
    for i in L:
        plt.plot([i + 1 for i in range(len(d2))], accur[i][:], label=model_name[i])

    plt.legend()
    plt.xlabel("Epochs")
    plt.title("Accuracy of Model")
    plt.show()
    plt.figure(3)
    for i in L:
        plt.plot([i for i in range(1, 1 + len(d1) * log_interval, log_interval)], loss[i][:], label=model_name[i])
    plt.legend()
    plt.xlabel("Times")
    plt.title("Loss of Model")
    plt.show()
