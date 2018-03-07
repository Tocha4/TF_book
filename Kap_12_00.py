import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import struct

#%% load_mnist function
def load_mnist(path, kind='train'):
    """MNIST-Daten von `path` laden"""
    labels_path = os.path.join(path,
    '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path,
    '%s-images.idx3-ubyte' % kind)
    print(images_path)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        print(magic, n)
        labels = np.fromfile(lbpath, dtype=np.uint8)
        
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        
    return images, labels

images, labels = load_mnist(r'..\mnist')


#%% batch_generator function


def batch_generator(X,y, batch_size=64, shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])
    
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
        
    for i in range(0,X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i: i+batch_size])