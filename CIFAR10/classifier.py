import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from numpy import linalg as LA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix
import time


def CIFAR10_Regressor(feature, train_labels, num_clusters, class_list, weights, bias, random_seed=1, print_detail=False):
    use_classes = len(class_list)
    for k in range(len(num_clusters)):
        if k != len(num_clusters) - 1:
            num_clus = int(num_clusters[k] / use_classes)
            labels = np.zeros((feature.shape[0], num_clusters[k]))
            for n in range(use_classes):
                idx = (train_labels == class_list[n])
                index = np.where(idx == True)[0]
                feature_special = np.zeros((index.shape[0], feature.shape[1]))
                for i in range(index.shape[0]):
                    feature_special[i] = feature[index[i]] # The largest Features = 1600: No need to use GPU KMeans
                # labels_, _ = kmeans(
                #     X=torch.from_numpy(feature_special).to("cuda"),
                #     num_clusters=num_clus, distance='euclidean', device=torch.device('cuda')
                # )
                kmeans = KMeans(n_clusters=num_clus, random_state=random_seed).fit(feature_special)
                pred_labels = kmeans.labels_
                for i in range(feature_special.shape[0]):
                    labels[index[i], pred_labels[i] + n * num_clus] = 1

            # least square regression
            A = np.ones((feature.shape[0], 1))
            feature = np.concatenate((A, feature), axis=1)
            weight = np.matmul(LA.pinv(feature), labels)
            feature = np.matmul(feature, weight)
            weights['%d LLSR weight' % k] = weight[1:weight.shape[0]]
            bias['%d LLSR bias' % k] = weight[0].reshape(1, -1)
            if (print_detail == True):
                print(k, ' layer LSR weight shape:', weight.shape)
                print(k, ' layer LSR output shape:', feature.shape)

            pred_labels = np.argmax(feature, axis=1)
            num_clas = np.zeros((num_clusters[k], use_classes))
            for i in range(num_clusters[k]):
                for t in range(use_classes):
                    for j in range(feature.shape[0]):
                        if pred_labels[j] == i and train_labels[j] == t:
                            num_clas[i, t] += 1
            acc_train = np.sum(np.amax(num_clas, axis=1)) / feature.shape[0]
            if(print_detail==True):
                print(k, ' layer LSR training acc is {}'.format(acc_train))

            # Relu
            for i in range(feature.shape[0]):
                for j in range(feature.shape[1]):
                    if feature[i, j] < 0:
                        feature[i, j] = 0
        else:
            # least square regression
            labels = nn.functional.one_hot(torch.from_numpy(train_labels.astype(np.int64)), len(class_list)).numpy()
            A = np.ones((feature.shape[0], 1))
            feature = np.concatenate((A, feature), axis=1)
            weight = np.matmul(LA.pinv(feature), labels)
            feature = np.matmul(feature, weight)
            weights['%d LLSR weight' % k] = weight[1:weight.shape[0]]
            bias['%d LLSR bias' % k] = weight[0].reshape(1, -1)
            if (print_detail == True):
                print(k, ' layer LSR weight shape:', weight.shape)
                print(k, ' layer LSR output shape:', feature.shape)

            pred_labels = np.argmax(feature, axis=1)
            acc_train = sklearn.metrics.accuracy_score(train_labels, pred_labels)
    return weights, bias, acc_train

def clf(dataset, feature, train_labels, class_list, random_seed, print_detail):
    starting_training_time = time.time()
    # feature normalization
    std_var = (np.std(feature, axis=0)).reshape(1, -1)
    feature = feature / std_var

    if (dataset != "MNIST" and dataset != "Tiny_ImageNet"):
        num_clusters = [200, 100, len(class_list)]
    elif (dataset == "MNIST"):
        num_clusters = [120, 84, len(class_list)]
    else:
        if(len(class_list) <= 10):
            num_clusters = [200, 100, len(class_list)]
        elif(len(class_list) <= 100):
            num_clusters = [400, 200, len(class_list)]
        else:
            num_clusters = [600, 400, len(class_list)]

    weights = {}
    bias = {}
    weight, bias, acc_train = CIFAR10_Regressor(feature, train_labels, num_clusters, class_list, weights, bias, random_seed, print_detail)

    ending_time_training = time.time()
    return weights, bias, acc_train, ending_time_training-starting_training_time



def testing_clf(dataset, feature, labels, weights, biases, class_list, print_detail):
    if (dataset != "MNIST"):
        num_clusters = [200, 100, len(class_list)]
    elif (dataset == "MNIST"):
        num_clusters = [120, 84, len(class_list)]

    # feature normalization
    std_var = (np.std(feature, axis=0)).reshape(1, -1)
    feature = feature / std_var

    for k in range(len(num_clusters)):
        weight = weights['%d LLSR weight' % k]
        bias = biases['%d LLSR bias' % k]
        feature = np.matmul(feature, weight) + bias
        if(print_detail==True):
            print(k, ' layer LSR weight shape:', weight.shape)
            print(k, ' layer LSR output shape:', feature.shape)
        if k != len(num_clusters) - 1:
            # Relu
            for i in range(feature.shape[0]):
                for j in range(feature.shape[1]):
                    if feature[i, j] < 0:
                        feature[i, j] = 0
        else:
            pred_labels = np.argmax(feature, axis=1)
            acc = sklearn.metrics.accuracy_score(labels, pred_labels)
            matrix = confusion_matrix(labels, pred_labels)


    return acc, matrix





