import torch
import torch.nn as nn
import numpy as np
import sklearn
from numpy import linalg as LA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix
import time


def CIFAR10_Regressor(feature, train_labels, class_list, weights, bias, print_detail=False):
    # least square regression
    labels = nn.functional.one_hot(torch.from_numpy(train_labels.astype(np.int64)), len(class_list)).numpy()
    A = np.ones((feature.shape[0], 1))
    feature = np.concatenate((A, feature), axis=1)
    weight = np.matmul(LA.pinv(feature), labels)
    feature = np.matmul(feature, weight)
    weights['%d LLSR weight'] = weight[1:weight.shape[0]]
    bias['%d LLSR bias'] = weight[0].reshape(1, -1)
    if (print_detail == True):
        print(' layer LSR weight shape:', weight.shape)
        print(' layer LSR output shape:', feature.shape)

    pred_labels = np.argmax(feature, axis=1)
    acc_train = sklearn.metrics.accuracy_score(train_labels, pred_labels)
    return weights, bias, acc_train
    # return acc_train

def simplified_clf(feature, train_labels, class_list, print_detail):
    starting_training_time = time.time()
    # feature normalization
    std_var = (np.std(feature, axis=0)).reshape(1, -1)
    feature = feature / std_var
    weights = {}
    bias = {}
    weight, bias, acc_train = CIFAR10_Regressor(feature, train_labels, class_list, weights, bias, print_detail)
    # acc_train = CIFAR10_Regressor(feature, train_labels, class_list, weights, bias, print_detail)

    ending_time_training = time.time()
    return weights, bias, acc_train, ending_time_training-starting_training_time
    # return acc_train, ending_time_training - starting_training_time


def simplified_testing_clf(feature, labels, weights, biases):

    # feature normalization
    std_var = (np.std(feature, axis=0)).reshape(1, -1)
    feature = feature / std_var

    weight = weights['%d LLSR weight']
    bias = biases['%d LLSR bias']
    feature = np.matmul(feature, weight) + bias

    pred_labels = np.argmax(feature, axis=1)
    acc = sklearn.metrics.accuracy_score(labels, pred_labels)
    matrix = confusion_matrix(labels, pred_labels)

    return acc, matrix


