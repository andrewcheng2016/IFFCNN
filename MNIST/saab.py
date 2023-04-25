import numpy as np
from skimage.util.shape import view_as_windows
import torch
from sklearn.decomposition import PCA, IncrementalPCA
from numpy import linalg as LA
from skimage.measure import block_reduce
import scipy

from IPCA import IPCA


# convert responses to patches representation
def window_process(samples, kernel_size, stride):
    '''
	Create patches
	:param samples: [num_samples, feature_height, feature_width, feature_channel]
	:param kernel_size: int i.e. patch size
	:param stride: int
	:return patches: flattened, [num_samples, output_h, output_w, feature_channel*kernel_size^2]
	'''
    n, h, w, c = samples.shape #(num of samples, height, width, channel)
    output_h = (h - kernel_size) // stride + 1
    output_w = (w - kernel_size) // stride + 1
    patches = view_as_windows(np.ascontiguousarray(samples), (1, kernel_size, kernel_size, c),
                              step=(1, stride, stride, c))
    patches = patches.reshape(n, output_h, output_w, c * kernel_size * kernel_size)
    return patches


def remove_mean(features, axis):
    '''
	Remove the dataset mean.
	:param features [num_samples,...]
	:param axis the axis to compute mean
	'''
    # print("features.shape:",features.shape)
    feature_mean = np.mean(features, axis=axis, keepdims=True)
    feature_remove_mean = features - feature_mean
    return feature_remove_mean, feature_mean


def select_balanced_subset(images, labels, use_num_images, use_classes, seed=0):
    '''
	select equal number of images from each classes
	'''
    # Shuffle
    np.random.seed(seed)
    num_total = images.shape[0]
    shuffle_idx = np.random.permutation(num_total)
    images = images[shuffle_idx]
    labels = labels[shuffle_idx]

    num_class = len(use_classes)
    num_per_class = int(use_num_images / num_class)
    selected_images = np.zeros((use_num_images, images.shape[1], images.shape[2], images.shape[3]))
    selected_labels = np.zeros(use_num_images)
    for i in range(num_class):
        # images_in_class=images[labels==i]
        idx = (labels == i)
        index = np.where(idx == True)[0]
        images_in_class = np.zeros((index.shape[0], images.shape[1], images.shape[2], images.shape[3]))
        for j in range(index.shape[0]):
            images_in_class[j] = images[index[j]]
        selected_images[i * num_per_class:(i + 1) * num_per_class] = images_in_class[:num_per_class]
        selected_labels[i * num_per_class:(i + 1) * num_per_class] = np.ones((num_per_class)) * i

    # Shuffle again
    np.random.seed(seed)
    shuffle_idx = np.random.permutation(num_per_class * num_class)
    selected_images = selected_images[shuffle_idx]
    selected_labels = selected_labels[shuffle_idx]

    return selected_images, selected_labels


def remove_zero_patch(samples, print_detail):
    std_var = (np.std(samples, axis=1)).reshape(-1, 1)
    ind_bool = (std_var == 0)
    ind = np.where(ind_bool == True)[0]
    if(print_detail == True):
        print('zero patch shape:', ind.shape)
    samples_new = np.delete(samples, ind, 0)
    return samples_new


def find_kernels_pca(training_data, num_kernels, energy_percent, pca_method, print_detail, batch_size=0, gpu_partition=10):
    '''
	Perform the PCA based on the provided samples.
	If num_kernels is not set, will use energy_percent.
	If neither is set, will preserve all kernels.
	:param samples: [num_samples, feature_dimension]
	:param num_kernels: num kernels to be preserved
	:param energy_percent: the percent of energy to be preserved
	:return: kernels
	'''

    if(print_detail == True):
        print("\nTraining data's shape: {}, Total size: {}".format(training_data.shape, training_data.size))
        print("Training data's size: {} Gbytes".format(training_data.size*training_data.itemsize/1e9))

    if(pca_method == "sklearn" or pca_method == "IPCA"):
        pca = PCA(n_components=training_data.shape[1], svd_solver='full', whiten=True)
        pca.fit(training_data)
        # Compute the number of kernels corresponding to preserved energy
        if energy_percent:
            energy = np.cumsum(pca.explained_variance_ratio_)
            num_components = np.sum(energy < energy_percent) + 1
        else:
            num_components = num_kernels

        kernels = pca.components_[:num_components, :]

        if(print_detail == True):
            print("Num of kernels: %d" % num_components)
            # print("Energy percent: %f" % np.cumsum(pca.explained_variance_ratio_)[num_components - 1])

    elif(pca_method == "sklearn_IPCA"):
        pca = IncrementalPCA(n_components=training_data.shape[1], batch_size=batch_size)
        pca.fit(training_data)
        # Compute the number of kernels corresponding to preserved energy
        if energy_percent:
            energy = np.cumsum(pca.explained_variance_ratio_)
            num_components = np.sum(energy < energy_percent) + 1
        else:
            num_components = num_kernels

        kernels = pca.components_[:num_components, :]


        if (print_detail == True):
            print("Num of kernels: %d" % num_components)
            # print("Energy percent: %f" % np.cumsum(pca.explained_variance_ratio_)[num_components - 1])

    elif(pca_method == "svd"):
        U_svd, S, principal_components = scipy.linalg.svd(training_data, full_matrices=False)
        explained_variance_ = (S ** 2) / (training_data.shape[0] - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        if energy_percent:
            energy = np.cumsum(explained_variance_ratio_)
            num_components = np.sum(energy < energy_percent) + 1
        else:
            num_components = num_kernels
        kernels = principal_components[:num_components, :]
        if (print_detail == True):
            print("Num of kernels: %d" % num_components)
            print("Energy percent: %f" % np.cumsum(explained_variance_ratio_)[num_components - 1])

    elif (pca_method == "GPU" or pca_method == "GPU_IPCA"):
        num_components = num_kernels

        # TODO: Pytorch Covariance Matrix -> Pytorch svd
        training_data = training_data - np.mean(training_data, axis=0)
        # print("\ntraining_data.shape:",training_data.shape)
        start_idx = 0
        partition_size = int(training_data.shape[0]//gpu_partition)

        for m in range(gpu_partition):
            training_data_slice = training_data[start_idx:start_idx+partition_size]
            start_idx += partition_size
            training_data_tensor = torch.from_numpy(training_data_slice.T).to(device='cuda')
            if(m == 0):
                covariance_matrix = torch.cov(training_data_tensor)
            else:
                covariance_matrix += torch.cov(training_data_tensor)
        covariance_matrix /= partition_size

        u, s, v = torch.svd(covariance_matrix)
        principal_components = v.cpu().numpy().T
        kernels = principal_components[:num_components, :]
    return kernels


def find_kernels_ipca(dataset, sample_patches, dc, num_kernels, Q, sqs, m, n, use_gpu, print_detail):
    '''
	Performs the IPCA based on the provided samples.
	If num_kernels is not set, will use energy_percent.
	If neither is set, will preserve all kernels.
	:param samples: [num_samples, feature_dimension]
	:param num_kernels: num kernels to be preserved
	:param energy_percent: the percent of energy to be preserved
	:return: kernels, sample_mean
	'''

    # Remove patch mean
    sample_patches_centered, dc_new = remove_mean(sample_patches, axis=1) # Mean of each sample
    dc = np.vstack([dc, dc_new])
    if(dataset == "CIFAR10"):
        sample_patches_centered = remove_zero_patch(sample_patches_centered, print_detail=print_detail)


    # Remove feature mean (Set E(X)=0 for each dimension)
    training_data, feature_expectation = remove_mean(sample_patches_centered, axis=0)

    principal_components, latent_new, Q, m, sqs, sigma_new, z_new, n, gamma = IPCA(Q, n, training_data, m, sqs, use_gpu)

    if (print_detail == True):
        print("The shape of principal_components: ", principal_components.shape)


    principal_components = principal_components.cpu().numpy().T
    latent_new = latent_new.cpu().numpy()

    Q_new = Q.cpu().numpy()
    m = m.cpu().numpy()
    sqs = sqs.cpu().numpy()
    sigma_new = sigma_new.cpu().numpy()
    z_new = z_new.cpu().numpy()
    n = int(n.cpu().item())
    gamma = gamma.cpu().numpy()

    explained_variance = 100 * latent_new / sum(latent_new)

    num_components = num_kernels
    pca_det = np.linalg.det(principal_components)
    kernels = principal_components[:num_components, :]  # The shape of pca.components_ is (12, 16); The shape of kernels is (12, 16)
    mean = np.mean(sample_patches_centered, axis=0)

    coeff_orthogonal_count = 0
    coeff_orthogonal_flag = []

    for v in range(principal_components.shape[-1]):
        for j in range(principal_components.shape[-1]):
            if (j != v):
                if (round(np.dot(principal_components[:, v], principal_components[:, j]), 5) == 0):
                    coeff_orthogonal_count += 1
                else:
                    flag = np.array([v, j])
                    coeff_orthogonal_flag.append(flag)

    if (print_detail == True):
        print("Num of kernels: %d" % num_components)
        print("Energy percent: %f" % np.cumsum(explained_variance)[num_components - 1])

    num_channels = sample_patches.shape[-1]
    largest_ev = [np.var(dc * np.sqrt(num_channels))]
    dc_kernel = 1 / np.sqrt(num_channels) * np.ones((1, num_channels)) / np.sqrt(largest_ev)
    # kernels = np.concatenate((dc_kernel, kernels), axis=0).astype(np.float32)
    kernels = np.concatenate((dc_kernel, kernels), axis=0)



    return kernels, mean, dc, Q_new, m, sqs, n, sigma_new, gamma, pca_det, z_new, sample_patches, principal_components[:num_components, :]




def multi_Saab_transform(dataset, images, labels, kernel_sizes, num_kernels, energy_percent, pca_method, print_detail,
                         layer_no, batch_size=0, gpu_partition=10, use_ipca=False):
    '''
	Do the PCA "training".
	:param images: [num_images, height, width, channel]
	:param labels: [num_images]
	:param kernel_sizes: list, kernel size for each stage,
	       the length defines how many stages conducted
	:param num_kernels: list the number of kernels for each stage,
	       the length should be equal to kernel_sizes.
	:param energy_percent: the energy percent to be kept in all PCA stages.
	       if num_kernels is set, energy_percent will be ignored.
    :param use_num_images: use a subset of train images
    :param use_classes: the classes of train images
    return: pca_params: PCA kernels and projected images
    '''

    sample_images, selected_labels = images, labels

    i = layer_no
    num_samples = sample_images.shape[0]
    num_layers = 1
    pca_params = {}
    pca_params['num_layers'] = num_layers
    pca_params['kernel_size'] = kernel_sizes


    current_kernel_sizes = kernel_sizes if isinstance(kernel_sizes,int) else kernel_sizes[i]
    if (print_detail == True):
        print('--------stage %d --------' % i)
    # Create patches
    sample_patches = window_process(sample_images, current_kernel_sizes, 1)  # overlapping
    h = sample_patches.shape[1]
    w = sample_patches.shape[2]
    # Flatten
    sample_patches = sample_patches.reshape([-1, sample_patches.shape[-1]])
    if(print_detail == True):
        print("\nsample_patches' shape: {}, Total size: {}".format(sample_patches.shape, sample_patches.size))
        print("sample_patches' size: {} Gbytes".format(sample_patches.size * sample_patches.itemsize / 1e9))

    # Compute PCA kernel
    if not num_kernels is None:
        num_kernel = num_kernels if isinstance(kernel_sizes,int) else num_kernels[i]


    sample_patches_centered, dc = remove_mean(sample_patches, axis=1) # Remove samples' mean

    if (dataset == "CIFAR10"):
        sample_patches_centered = remove_zero_patch(sample_patches_centered, print_detail) # Remove patches that have zero std.

    # Remove feature mean (Set E(X)=0 for each dimension)
    training_data, feature_expectation = remove_mean(sample_patches_centered, axis=0)
    if(use_ipca == True):
        dc_mean = dc
        sqs = np.sum(np.square(training_data), axis=0)
        std = np.std(training_data, axis=0, ddof=1)
        Q = np.matmul(training_data.T, training_data)
        n = len(training_data)
        mean = np.mean(training_data, axis=0)
        pca_params['Layer_%d/mean' % i] = mean
        pca_params['Layer_%d/Q' % i] = Q
        pca_params['Layer_%d/n' % i] = n
        pca_params['Layer_%d/dc_mean' % i] = dc_mean
        pca_params['Layer_%d/sqs' % i] = sqs
        pca_params['Layer_%d/std' % i] = std

    num_channels = sample_patches.shape[-1]
    largest_ev = [np.var(dc * np.sqrt(num_channels))]

    # dc_kernel = (1 / np.sqrt(num_channels) * np.ones((1, num_channels)) / np.sqrt(largest_ev)).astype(np.float32)
    dc_kernel = (1 / np.sqrt(num_channels) * np.ones((1, num_channels)) / np.sqrt(largest_ev))

    kernels = find_kernels_pca(training_data, num_kernel,
                               energy_percent, pca_method,
                               print_detail=print_detail,
                               batch_size=batch_size,
                               gpu_partition=gpu_partition
                               )
    pc = kernels
    kernels = np.concatenate((dc_kernel, kernels), axis=0)
    num_channels = sample_patches.shape[-1]
    if i == 0:
        # Transform to get data for the next stage
        transformed = np.matmul(sample_patches, np.transpose(kernels))
    else:
        # Compute bias term
        bias = LA.norm(sample_patches, axis=1)
        bias = np.max(bias)
        pca_params['Layer_%d/bias' % i] = bias
        # Add bias
        sample_patches_centered_w_bias = sample_patches + 1 / np.sqrt(num_channels) * bias
        # Transform to get data for the next stage
        transformed = np.matmul(sample_patches_centered_w_bias, np.transpose(kernels))
        # Remove bias
        e = np.zeros((1, kernels.shape[0]))
        e[0, 0] = 1
        transformed -= bias * e

    # Reshape: place back as a 4-D feature map
    sample_images = transformed.reshape(num_samples, h, w, -1)

    # Maxpooling
    sample_images = block_reduce(sample_images, (1, 2, 2, 1), np.max)


    if(print_detail == True):
        print('Sample patches shape after flatten:', sample_patches.shape)
        print('Kernel shape:', kernels.shape)
        print('Transformed shape:', transformed.shape)
        print('Sample images shape:', sample_images.shape)

    pca_params['Layer_%d/kernel' % i] = kernels
    pca_params['Layer_%d/sample_patches' % i] = sample_patches
    pca_params['Layer_%d/dc' %i] = dc
    pca_params['Layer_%d/pc' % i] = pc


    return pca_params, sample_images


def multi_Saab_transform_IPCA(dataset, images, labels, kernel_sizes, num_kernels, pca_params, use_num_images, use_classes, use_gpu,
                              random_seed, print_detail, layer_no):
    '''
	Performs the PCA "training".
	:param images: [num_images, height, width, channel]
	:param labels: [num_images]
	:param kernel_sizes: list, kernel size for each stage,
	       the length defines how many stages conducted
	:param num_kernels: list the number of kernels for each stage,
	       the length should be equal to kernel_sizes.
	:param energy_percent: the energy percent to be kept in all PCA stages.
	       if num_kernels is set, energy_percent will be ignored.
    :param use_num_images: use a subset of train images
    :param use_classes: the classes of train images
    return: pca_params: PCA kernels and projected images
    '''

    num_total_images = images.shape[0]

    if use_num_images < num_total_images and use_num_images > 0:
        sample_images, selected_labels = select_balanced_subset(images, labels, use_num_images, use_classes,
                                                                random_seed)
    else:
        sample_images = images

    i = layer_no
    current_kernel_sizes = kernel_sizes if isinstance(kernel_sizes, int) else kernel_sizes[i]
    if (print_detail == True):
        print('--------stage %d --------' % i)

    Q = pca_params['Layer_%d/Q' % i]
    sqs = pca_params['Layer_%d/sqs' % i]
    dc_mean = pca_params['Layer_%d/dc_mean' % i]
    mean = pca_params['Layer_%d/mean' % i]
    n = pca_params['Layer_%d/n' % i]

    # Create patches
    sample_patches = window_process(sample_images, current_kernel_sizes, 1)  # overlapping
    h = sample_patches.shape[1]
    w = sample_patches.shape[2]
    # Flatten
    sample_patches = sample_patches.reshape([-1, sample_patches.shape[-1]])

    # Compute PCA kernel
    if not num_kernels is None:
        num_kernel = num_kernels if isinstance(kernel_sizes, int) else num_kernels[i]


    kernels, mean, dc, Q_new, m, sqs, n, std, gamma, pca_det, z_new, sample_patches, pc = find_kernels_ipca(dataset, sample_patches,
                                                                                                            dc_mean, num_kernel, Q, sqs,
                                                                                                            mean, n, use_gpu, print_detail=print_detail)

    num_channels = sample_patches.shape[-1]
    if i == 0:
        # Transform to get data for the next stage
        transformed = np.matmul(sample_patches, np.transpose(kernels))
    else:
        # Compute bias term
        bias = LA.norm(sample_patches, axis=1)
        bias = np.max(bias)
        pca_params['Layer_%d/bias' % i] = bias
        # Add bias
        sample_patches_centered_w_bias = sample_patches + 1 / np.sqrt(num_channels) * bias
        # Transform to get data for the next stage
        transformed = np.matmul(sample_patches_centered_w_bias, np.transpose(kernels))
        # Remove bias
        e = np.zeros((1, kernels.shape[0]))
        e[0, 0] = 1
        transformed -= bias * e

    # Reshape: place back as a 4-D feature map
    sample_images = transformed.reshape(len(sample_images), h, w, -1)

    # Maxpooling
    sample_images = block_reduce(sample_images, (1, 2, 2, 1), np.max)
    if (print_detail == True):
        print('Sample patches shape after flatten:', sample_patches.shape)
        print('Kernel shape:', kernels.shape)
        print('Transformed shape:', transformed.shape)
        print('Sample images shape:', sample_images.shape)

    pca_params['Layer_%d/kernel' % i] = kernels
    pca_params['Layer_%d/dc_mean' % i] = dc_mean
    pca_params['Layer_%d/Q' % i] = Q
    pca_params['Layer_%d/n' % i] = n
    pca_params['Layer_%d/mean' % i] = mean
    pca_params['Layer_%d/sqs' % i] = sqs
    pca_params['Layer_%d/std' % i] = std
    pca_params['Layer_%d/pc' % i] = pc

    return pca_params, sample_images



# Extract Features
def initialize(sample_images, layer_no, pca_params, print_detail=False):
    kernel_sizes = pca_params['kernel_size']
    i = layer_no

    if (print_detail == True):
        print('--------stage %d --------' % i)
    # Extract parameters
    kernels = pca_params['Layer_%d/kernel' % i]

    current_kernel_sizes = kernel_sizes if isinstance(kernel_sizes, int) else kernel_sizes[i]

    # Create patches
    sample_patches = window_process(sample_images,current_kernel_sizes, 1)  # overlapping
    h = sample_patches.shape[1]
    w = sample_patches.shape[2]
    # Flatten
    sample_patches = sample_patches.reshape([-1, sample_patches.shape[-1]])


    num_channels = sample_patches.shape[-1]
    if i == 0:
        # Transform to get data for the next stage
        transformed = np.matmul(sample_patches, np.transpose(kernels))
    else:
        bias = pca_params['Layer_%d/bias' % i]
        # Add bias
        sample_patches_centered_w_bias = sample_patches + 1 / np.sqrt(num_channels) * bias
        # Transform to get data for the next stage
        transformed = np.matmul(sample_patches_centered_w_bias, np.transpose(kernels))
        # Remove bias
        e = np.zeros((1, kernels.shape[0]))
        e[0, 0] = 1
        transformed -= bias * e

    # Reshape: place back as a 4-D feature map
    num_samples = sample_images.shape[0]
    sample_images = transformed.reshape(num_samples, h, w, -1)

    # Maxpooling
    sample_images = block_reduce(sample_images, (1, 2, 2, 1), np.max) # (400, 14, 14, 6)

    if(print_detail == True):
        print('Sample patches shape after flatten:', sample_patches.shape)
        print('Kernel shape:', kernels.shape)
        print('Transformed shape:', transformed.shape)
        print('Sample images shape:', sample_images.shape)

    return sample_images
