import numpy as np
import torchvision
import warnings


warnings.filterwarnings("ignore")

def get_data_for_class(images, labels, cls):
	if type(cls)==list:
		idx=np.zeros(labels.shape, dtype=bool)
		for c in cls:
			idx=np.logical_or(idx, labels==c)
	else:
		idx=(labels==cls)
	return images[idx], labels[idx]


def import_data():
    train_set = torchvision.datasets.CIFAR10('../../../../../Dataset/CIFAR10', train=True, download=True,
                                             transform=None)

    test_set = torchvision.datasets.CIFAR10('../../../../../Dataset/CIFAR10', train=False, download=True,
                                            transform=None)

    train_images = np.float32(train_set.data)/255
    train_labels = np.array(train_set.targets)
    test_images = np.float32(test_set.data)/255
    test_labels = np.array(test_set.targets)


    class_list = np.unique(train_labels).astype(np.int32).tolist()

    print("----------Finish loading CIFAR10 dataset----------")
    print("Shape of train_images:", train_images.shape)
    print("Shape of train_labels:", train_labels.shape)
    print("Shape of test_images:", test_images.shape)
    print("Shape of test_labels:", test_labels.shape)

    return train_images, train_labels, test_images, test_labels, class_list
