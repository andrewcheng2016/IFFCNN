import os
import shutil
import itertools
import numpy as np
import matplotlib.pyplot as plt



"""
To parse the class string.
"""
def parse_list_string(list_string):
	"""Convert the class string to list."""
	elem_groups=list_string.split(",")
	results=[]
	for group in elem_groups:
		term=group.split("-")
		if len(term)==1:
	  		results.append(int(term[0]))
		else:
	  		start=int(term[0])
	  		end=int(term[1])
	  		results+=range(start, end+1)
	return results


"""
To shuffle the dataset.
"""
def shuffle(images, labels, seed):
    np.random.seed(seed)
    if(images.ndim != 4):
        for i in range(images.shape[0]):
            num_total = images[i].shape[0]
            shuffle_idx = np.random.permutation(num_total)
            images[i] = images[i][shuffle_idx]
            labels[i] = labels[i][shuffle_idx]

    else:
        # Shuffle
        num_total = images.shape[0]
        shuffle_idx = np.random.permutation(num_total)
        images = images[shuffle_idx]
        labels = labels[shuffle_idx]

    return images, labels


"""
To get the index of each classes in testset so that the testset can be splitted according to the classes.
"""
def dataset_split(dataset, labelset, class_list):
    idx = []
    idx_list = []
    for i in range(len(class_list)):
        count = 0
        for m in range(labelset.size):
            if (labelset[m] == class_list[i]):
                idx.append(m)
                count += 1
        idx_list.append(count)
    dataset_sorted = dataset[idx]
    labelset_sorted = labelset[idx]

    return dataset_sorted, labelset_sorted, idx_list

"""
To split the dataset into batches.
"""
def dataset_batch(dataset, labels, total_stage, batch_size, random_seed):
    images_batch = []
    labels_batch = []
    # Distribute the sorted images of different classes in different batches
    start = 0
    for m in range(total_stage):
        images_batch.append(dataset[start:start + batch_size])
        labels_batch.append(labels[start:start + batch_size])
        start += batch_size

    images_batch, labels_batch = np.array(images_batch).reshape(total_stage, batch_size, dataset.shape[1], dataset.shape[2], dataset.shape[3]),\
                                 np.array(labels_batch).reshape(total_stage, batch_size)

    images_batch, labels_batch = shuffle(images_batch, labels_batch, random_seed)

    return np.array(images_batch), np.array(labels_batch)


"""
To remove the path and create a new one.
"""
def path_check(save_path, remove_path=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if(os.path.exists(save_path) and remove_path == True):
        shutil.rmtree(save_path)
        os.makedirs(save_path)



"""
To plot the confusion matrix.
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix is saved")
    else:
        print('Confusion matrix, without normalization is saved')

    #print(cm)

    if(len(classes) <= 20):
        plt.figure(figsize=(10, 8))
        plt.title(title, fontsize=16)
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)
    elif(len(classes) <= 40):
        plt.figure(figsize=(30, 28))
        plt.title(title, fontsize=24)
        plt.ylabel('True label', fontsize=24)
        plt.xlabel('Predicted label', fontsize=24)
    elif (len(classes) <= 60):
        plt.figure(figsize=(50, 38))
        plt.title(title, fontsize=32)
        plt.ylabel('True label', fontsize=32)
        plt.xlabel('Predicted label', fontsize=32)
    else:
        plt.figure(figsize=(70, 58))
        plt.title(title, fontsize=64)
        plt.ylabel('True label', fontsize=64)
        plt.xlabel('Predicted label', fontsize=64)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.ylabel('True label', fontsize=18)
    # plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()


"""
To plot the comparison of different time.
"""
def comparisonPlot(opt, time, method, timeType, remove_path=False):
    plt.figure(figsize=(10, 8))
    # plt.title('Comparison of {} time'.format(timeType), fontsize=16)
    plt.ylabel('Time (s)', fontweight='bold')
    plt.xlabel('Number of samples', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(True)
    plt.rcParams['font.weight'] = 'bold'
    color=['blue', 'green', 'purple', 'red']
    x = np.arange(opt.num_samples_per_batch, opt.num_samples_per_batch*(time.shape[1]+1), opt.num_samples_per_batch)
    for l in range(len(method)):
        plt.plot(x, time[l], label=method[l], color=color[l], marker='o')
        plt.legend(loc='upper left', fontsize=16)

    save_path = './Comparison/BatchSize_{}/'.format(opt.num_samples_per_batch)
    path_check(save_path, remove_path=remove_path)
    plt.savefig(save_path + '/{}.png'.format(timeType))


