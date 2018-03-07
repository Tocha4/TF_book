import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from Kap_12_00 import load_mnist, batch_generator


X_data, y_data = load_mnist('../mnist/', kind='train')

X_test, y_test = load_mnist('../mnist/', kind='t10k')

X_train, y_train = X_data[:50000,:], y_data[:50000]
X_valid, y_valid = X_data[50000:,:], y_data[50000:]

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train-mean_vals)/std_val
X_test_centered = (X_test-mean_vals)/std_val

def conv_layer(input_tensor, name, kernel_size, n_output_channels, padding_mode='SAME', strides=(1,1,1,1)):
    
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]
        weights_shape = list(kernel_size) + [n_input_channels, n_output_channels]
        weights = tf.get_variable(name='_weights',shape=weights_shape)
        
        print(1, weights)        
        biases = tf.get_variable(name='_biases',initializer=tf.zeros(shape=[n_output_channels]))        
        print(2, biases)        
        conv = tf.nn.conv2d(input=input_tensor, filter=weights, strides=strides, padding=padding_mode)
        print(3, conv)
        conv = tf.nn.bias_add(conv, biases, name='net_pre-activation')
        print(4, conv)
        conv = tf.nn.relu(conv, name='activation')
        print(5, conv)
        
        return conv
    
    
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28,28,1]) #[batch, width, height, channels (RGB or Grey)]
    conv_layer(x, name='convtest', kernel_size=(3,3), n_output_channels=32)
    
del g,x