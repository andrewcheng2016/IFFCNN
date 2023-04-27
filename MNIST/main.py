import datetime
from time import time
import numpy as np
import random
import argparse
import logging
import matplotlib.pyplot as plt

import data

from FFCNN import get_kernel, get_kernel_online, get_feature
from Toolbox import parse_list_string, dataset_batch, path_check, plot_confusion_matrix
from classifier import clf, testing_clf

def parse_arg():
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s]: %(levelname)s: %(message)s"
    )

    # General Parameters
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-dataset', choices=['MNIST', 'CIFAR10'], default='MNIST')
    parser.add_argument('-num_class_per_batch', type=int, default=2) # Num of classes per batch
    parser.add_argument('-random_seed', type=int, default=1) # Random Seed for shuffling and model initialization
    parser.add_argument('-channel', type=int, default=1)
    parser.add_argument('-resolution', type=int, default=32)

    # Parameters of the FFCNN
    parser.add_argument('-kernel_sizes', choices=["3,3,3", "5,5"], default="5,5")  # Kernels size for each stage
    parser.add_argument('-num_kernels', choices=["5,15", "31,63,127"], default="5,15")  # num_kernels = "31,63" if dataset != "MNIST" else "5,15"
    parser.add_argument('-energy_percent', type=float, default=None)  # Energy to be preserved in each stage
    parser.add_argument('-num_samples_per_batch', type=int, default=10000) # Num of new samples per batch
    parser.add_argument('-PCA_method_1st', choices=["sklearn", "svd", "GPU", "IPCA", "GPU_IPCA"], default="GPU_IPCA")  # Methods to perform pca at 1st layer
    parser.add_argument('-PCA_method_2nd', choices=["sklearn", "svd", "GPU", "IPCA", "GPU_IPCA"], default="GPU_IPCA")  # Methods to perform pca at 2nd layer
    parser.add_argument('-gpu_partition', type=int, default=5) # Num of partitions for GPU IPCA
    parser.add_argument('-print_detail', action='store_true', default=False)  # Print details of the process?

    opt = parser.parse_args()
    return opt


def main():
    opt = parse_arg()
    dataset = opt.dataset
    random_seed = opt.random_seed

    # read data
    train_images, train_labels, test_images, test_labels, class_list = data.import_data(opt)
    train_images, test_images = train_images.astype(np.float32), test_images.astype(np.float32)
    n_class = len(np.unique(train_labels))
    print_detail = opt.print_detail
    num_samples_per_batch = opt.num_samples_per_batch if opt.num_samples_per_batch != -1 else len(train_images)

    total_images = len(train_images)
    total_stage = int(total_images / num_samples_per_batch)


    # Get num_class_per_batch classes for each class for each batch
    train_images_batch, train_labels_batch = dataset_batch(train_images, train_labels, total_stage, num_samples_per_batch, random_seed)

    use_ipca = True if "IPCA" in opt.PCA_method_1st or "IPCA" in opt.PCA_method_2nd else False

    device = "GPU" if "GPU" in opt.PCA_method_1st or "GPU" in opt.PCA_method_2nd else "CPU"


    print("Dataset Generation Done!")

    random.seed(random_seed)
    kernel_sizes = parse_list_string(opt.kernel_sizes)
    num_kernels = opt.num_kernels
    num_kernels = parse_list_string(num_kernels) if num_kernels else None
    energy_percent = opt.energy_percent
    gpu_partition = opt.gpu_partition if device == "GPU" else 0

    print('\n----------General Parameters----------')
    print("Dataset: %s" % (dataset))
    print('\nRandom Seed:', random_seed)
    print('Number of samples per batch:', opt.num_samples_per_batch)
    print('Number of stages:', total_stage)
    print('Total Number of classes:', n_class)

    print('\n----------FFCNN Parameters----------')
    print('Kernel_sizes:', kernel_sizes)
    print('Number_kernels:', num_kernels)
    print('Energy_percent:', energy_percent)
    print('1st layer PCA method:', opt.PCA_method_1st)
    print('2nd layer PCA method:', opt.PCA_method_2nd)
    print('Use IPCA:', use_ipca)
    print('Device:', device)
    if (device == "GPU"):
        print('GPU partition:', gpu_partition)
    print('Print IFFCNN detail:', print_detail)

    stage_time = []
    stage_train_acc = []
    stage_test_acc = []
    get_kernel_time = []
    get_feature_time = []
    classifier_training_time = []

    starting_time_whole_process = time()
    for stage in range(total_stage):
        start_time_stage = time()
        images = train_images_batch[stage].astype(np.float32)
        labels = train_labels_batch[stage]

        if (stage == 0):
            trained_images = train_images_batch[stage]
            trained_labels = train_labels_batch[stage]
        else:
            trained_images = np.concatenate((trained_images, train_images_batch[stage]), axis=0)
            trained_labels = np.append(trained_labels, train_labels_batch[stage])



        trained_images_num = len(trained_images)
        print("\n\n==================== [Samples:{}/{}] (1st layer:{} / 2nd layer:{} / Device:{}) ====================".format(trained_images_num,
                                                                                                                              total_images,
                                                                                                                              opt.PCA_method_1st,
                                                                                                                              opt.PCA_method_2nd,
                                                                                                                              device))
        print("\nStarting time for the process: {} ".format(datetime.datetime.now()))


        use_classes = np.unique(train_labels).astype(np.int32)

        start_getKernel_time = time()
        if use_ipca == True and stage != 0:
            pca_params = get_kernel_online(dataset, images, labels, kernel_sizes, num_kernels, pca_params,
                                           pca_method_1st=opt.PCA_method_1st, pca_method_2nd=opt.PCA_method_2nd,
                                           use_classes=use_classes.tolist(), random_seed=opt.random_seed,
                                           print_detail=opt.print_detail)



        else:
            print("Number of trained images: ", len(trained_images))
            pca_params = get_kernel(dataset, trained_images, trained_labels, kernel_sizes, num_kernels,
                                    opt.energy_percent,
                                    pca_method_1st=opt.PCA_method_1st, pca_method_2nd=opt.PCA_method_2nd,
                                    gpu_partition=gpu_partition,
                                    use_ipca=use_ipca,
                                    print_detail=opt.print_detail)

        end_getKernel_time = time()
        get_kernel_time.append(end_getKernel_time - start_getKernel_time)

        # flops = high.stop_counters()
        pca_params_list = []
        for m in range(len(num_kernels)):
            print("Creating pca kernels list: pca_params[PCA Kernels/Layer{}]".format(m))
            pca_params_list.append(pca_params['PCA Kernels/Layer{}'.format(m)])

        start_getFeature_time = time()
        _, train_feature = get_feature(trained_images, pca_params_list, trainset=True, print_detail=print_detail)
        end_getFeature_time = time()
        get_feature_time.append(end_getFeature_time - start_getFeature_time)

        _, test_feature = get_feature(test_images, pca_params_list, trainset=False, print_detail=print_detail)


        # TODO: Start Training Model
        print('--------Start training the classifier--------')
        print("Type of train_feature: ", train_feature.dtype)
        weights, biases, train_acc, training_time = clf(dataset, train_feature, trained_labels,
                                                        use_classes, random_seed, print_detail=print_detail)
        classifier_training_time.append(training_time)
        print('--------Finish training the classifier--------')

        test_acc, confusion_matrix = testing_clf(dataset, test_feature, test_labels, weights, biases,
                                                 use_classes, print_detail=print_detail)

        stage_train_acc.append(train_acc)
        stage_test_acc.append(test_acc)
        end_time_stage = time()
        stage_time.append(end_time_stage - start_time_stage)

        print('[Samples:{}/{}] Prediction Training acc (1st layer:{} / 2nd layer:{} / Device:{}) is {:.2%}'.format(trained_images_num, total_images,
                                                                                                                 opt.PCA_method_1st,
                                                                                                                 opt.PCA_method_2nd,
                                                                                                                 device, train_acc))
        print('[Samples:{}/{}] Prediction Testing acc (1st layer:{} / 2nd layer:{} / Device:{}) is {:.2%}'.format(trained_images_num, total_images,
                                                                                                                opt.PCA_method_1st,
                                                                                                                opt.PCA_method_2nd,
                                                                                                                device, test_acc))

        print('[Samples:{}/{}] Time Used (1st layer:{} / 2nd layer:{} / Device:{}) is {}'.format(trained_images_num,
                                                                                               total_images,
                                                                                               opt.PCA_method_1st,
                                                                                               opt.PCA_method_2nd,
                                                                                               device,
                                                                                               datetime.timedelta(seconds=end_time_stage - start_time_stage)))

    ending_time_whole_process = time()
    remove_path = True
    plot_confusion_matrix(confusion_matrix, classes=class_list, normalize=True, title='Confusion Matrix')
    used_features = int(train_feature.shape[-1])
    save_path = './Confusion_Matrix/{}'.format(opt.PCA_method_1st)
    path_check(save_path, remove_path=remove_path)
    plt.savefig(save_path + '/{}_(Class{}-{}).png'.format(dataset, int(class_list[0]),
                                                          int(class_list[-1])))

    print("\n==================== Time Summary (1st layer:{} / 2nd layer:{} / Device:{}) ====================".format(opt.PCA_method_1st,
                                                                                                                    opt.PCA_method_2nd,
                                                                                                                    device))

    trained_images_num = 0
    for stage in range(total_stage):
        trained_images_num += num_samples_per_batch
        print("Stage {} [Samples:{}/{}] Time used: {}".format(stage, trained_images_num, len(train_images),
                                                              datetime.timedelta(seconds=stage_time[stage])))

    print("\n")
    trained_images_num = 0
    for stage in range(total_stage):
        trained_images_num += num_samples_per_batch
        print("Stage {} [Samples:{}/{}] Time used for getting kernel: {}".format(stage, trained_images_num, len(train_images),
                                                                                 datetime.timedelta(seconds=get_kernel_time[stage])))

    print("\n")
    trained_images_num = 0
    for stage in range(total_stage):
        trained_images_num += num_samples_per_batch
        print("Stage {} [Samples:{}/{}] Time used for getting feature: {}".format(stage, trained_images_num, len(train_images),
                                                                                  datetime.timedelta(seconds=get_feature_time[stage])))

    print("\n")
    trained_images_num = 0
    for stage in range(total_stage):
        trained_images_num += num_samples_per_batch
        print("Stage {} [Samples:{}/{}] Time used for training classifier: {}".format(stage, trained_images_num, len(train_images),
                                                                                      datetime.timedelta(seconds=classifier_training_time[stage])))




    print("\nTime used for the whole process: {} ".format(
        datetime.timedelta(seconds=ending_time_whole_process - starting_time_whole_process)))
    print("Ending time for the whole process: {}".format(datetime.datetime.now()))

    print("\n==================== Accuracy Summary (1st layer:{} / 2nd layer:{} / Device:{}) ====================".format(opt.PCA_method_1st,
                                                                                                                        opt.PCA_method_2nd,
                                                                                                                        device))

    trained_images_num = 0
    for stage in range(total_stage):
        trained_images_num += num_samples_per_batch
        print("Stage {} [Samples:{}/{}] Train acc: {:.2%} Test acc: {:.2%}".format(stage, trained_images_num,
                                                                                   len(train_images),
                                                                                   stage_train_acc[stage],
                                                                                   stage_test_acc[stage]))

    print("\nTrain accuracy: {:.2%}".format(train_acc))
    print("Test accuracy: {:.2%}".format(test_acc))

    print("\n\n==================== End of (1st layer:{} / 2nd layer:{} / Device:{}) ====================".format(opt.PCA_method_1st,
                                                                                                                opt.PCA_method_2nd,
                                                                                                                device))


if __name__ == '__main__':
	main()

