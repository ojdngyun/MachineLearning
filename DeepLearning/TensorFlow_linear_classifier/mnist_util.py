import urllib.request
import os
import gzip
import numpy as np
from loader import MNIST

url = 'url'
filename = 'filename'
files = {
    'train_images': {
        url: 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        filename: 'train-images-idx3-ubyte.gz'
    },
    'train_labels': {
        url: 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        filename: 'train-labels-idx1-ubyte.gz'
    },
    'test_images': {
        url: 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        filename: 't10k-images-idx3-ubyte.gz'
    },
    'test_labels': {
        url: 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
        filename: 't10k-labels-idx1-ubyte.gz',
    }
}


def download_data(download_url, file_path):
    urllib.request.urlretrieve(
        download_url,
        file_path
    )


def convert_one_hot(labels):
    one_hot_labels = []
    for label in labels:
        zeros = np.zeros(10)
        zeros[label] = 1
        one_hot_labels.append(zeros)
    return np.array(one_hot_labels)


def load_mnist_file(directory_path=None, one_hot=True):
    if directory_path is None:
        print('Error! no path found.')
        return
    current_files = {file for file in os.listdir(directory_path)}
    for file in files.values():
        if file[filename] not in current_files:
            print(file[filename], ' not in local directory')
            path = directory_path + '/' + file[filename]
            download_data(file[url], path)
            file_in = gzip.open(path, 'rb')
            file_out = open(path.replace('.gz', ''), 'wb')
            file_out.write(file_in.read())
            file_out.close()
            file_in.close()

    mndata = MNIST(directory_path, return_type='numpy')
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    if one_hot:
        train_labels = convert_one_hot(train_labels)
        test_labels = convert_one_hot(test_labels)

    return train_images, train_labels, test_images, test_labels
