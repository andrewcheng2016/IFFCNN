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


def import_data(opt):
	train_set = torchvision.datasets.MNIST('../../../../../Dataset/MNIST', train=True, download=True, transform=None)
	test_set = torchvision.datasets.MNIST('../../../../../Dataset/MNIST', train=False, download=True, transform=None)

	# train_images = np.float32(train_set.train_data.unsqueeze(-1).numpy())
	train_images = train_set.train_data.unsqueeze(-1).numpy()/255
	train_labels = train_set.train_labels.numpy()
	# test_images = np.float32(test_set.test_data.unsqueeze(-1).numpy())
	test_images = test_set.test_data.unsqueeze(-1).numpy()/255
	test_labels = test_set.test_labels.numpy()

	# zeropadding
	train_images = np.pad(train_images, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
	test_images = np.pad(test_images, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')

	train_images = train_images.reshape(-1, opt.resolution, opt.resolution, 1)
	test_images = test_images.reshape(-1, opt.resolution, opt.resolution, 1)

	class_list = np.unique(train_labels).astype(np.int32).tolist()


	print("----------Finish loading MNIST dataset----------")
	print("Shape of train_images:", train_images.shape)
	print("Shape of train_labels:", train_labels.shape)
	print("Shape of test_images:", test_images.shape)
	print("Shape of test_labels:", test_labels.shape)

	return train_images, train_labels, test_images, test_labels, class_list
