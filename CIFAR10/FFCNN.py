import sys
import numpy as np
from tqdm import tqdm

import saab

def get_kernel(dataset, train_images, train_labels, kernel_sizes, num_kernels, energy_percent,
               pca_method_1st, pca_method_2nd, gpu_partition=10, use_ipca=False, print_detail=True):

    pca_params_list = {}
    for m in tqdm(range(len(num_kernels)), desc="Calculating PCA Kernels", file=sys.stdout):
        pca_method = pca_method_1st if m == 0 else pca_method_2nd
        pca_params_list['PCA Kernels/Layer{}'.format(m)], train_images, = saab.multi_Saab_transform(dataset, train_images, train_labels,
                                                                                                    kernel_sizes=kernel_sizes[m],
                                                                                                    num_kernels=num_kernels[m],
                                                                                                    energy_percent=energy_percent,
                                                                                                    pca_method=pca_method,
                                                                                                    print_detail=print_detail,
                                                                                                    layer_no=m,
                                                                                                    gpu_partition=gpu_partition,
                                                                                                    use_ipca=use_ipca)
    return pca_params_list


def get_kernel_online(dataset, train_images, train_labels, kernel_sizes, num_kernels, old_pca_params,
                      pca_method_1st, pca_method_2nd, use_classes, random_seed, energy_percent, print_detail=True):

    pca_params_list = {}
    for m in range(len(num_kernels)):
            print("Calculating IPCA Kernels for Layer {}".format(m))
            pca_method = pca_method_1st if m == 0 else pca_method_2nd
            use_gpu = True if "GPU" in pca_method else False
            if (m != 0):
                train_images, _ = get_feature(train_images, [pca_params_list['PCA Kernels/Layer{}'.format(m - 1)]],
                                              trainset=True, print_detail=print_detail)
                train_images = train_images.astype(np.float32)



            pca_params, _ = saab.multi_Saab_transform_IPCA(dataset, train_images, train_labels,
                                                           kernel_sizes=kernel_sizes[m],
                                                           num_kernels=num_kernels[m],
                                                           pca_params=old_pca_params['PCA Kernels/Layer{}'.format(m)],
                                                           use_num_images=train_images.shape[0],
                                                           use_classes=use_classes,
                                                           random_seed=random_seed,
                                                           print_detail=print_detail,
                                                           layer_no=m,
                                                           use_gpu=use_gpu
                                                           )
            pca_params_list['PCA Kernels/Layer{}'.format(m)] = pca_params
    return pca_params_list

def get_feature(images, pca_params_list, trainset=True, print_detail=False):
    datatype = "Trainset" if(trainset == True) else "Testset"
    for i in range(len(pca_params_list)):
        if(print_detail == True):
            print("Extracting features for {} Layer {}".format(datatype, i))
        images = saab.initialize(images, i, pca_params_list[i])
    features = images.reshape(len(images), -1)
    return images, features







