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

#%% functions
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
    
    
def fc_layer(input_tensor, name, n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        print('Input_Units',n_input_units)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))
        
        weights_shape = [n_input_units, n_output_units]
        weights = tf.get_variable(name='_weights', shape=weights_shape)
        print('weights:', weights)
        biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[n_output_units]))
        print('biases:', biases)
        layer = tf.matmul(input_tensor, weights)
        print('layer:', layer)
        if activation_fn is None:
            return layer
        layer = activation_fn(layer, name='activation')
        print('layer_2:', layer)
        return layer
   
def build_cnn():
    tf_x = tf.placeholder(tf.float32, shape=[None, 784], name='tf_x')
    # Images come in as flatted array in a batch: Shape=[batch, 784]
    tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')    
    tf_x_image = tf.reshape(tf_x, shape=[-1,28,28,1], name='tf_reshaped')
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32, name='tf_y_onehot')
        
    
    print('\nErste Schicht: Faltung_1')
    # 1. Faltungsschicht
    h1 = conv_layer(tf_x_image, name= 'conv_1', kernel_size=(5,5), n_output_channels=32, padding_mode='VALID')
    # 1. Max-Pooling
    h1_pool = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    print('\nZweite Schicht: Faltung_2')
    # 2. Faltungsschicht
    h2 = conv_layer(h1_pool, name= 'conv_2', kernel_size=(5,5), n_output_channels=64, padding_mode='VALID')
    # 2. Max-Pooling
    h2_pool = tf.nn.max_pool(h2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    print('\nDritte Schicht: Vollständig verknüpft')
    # 3. Schicht
    h3 = fc_layer(h2_pool, name= 'fc_3', n_output_units=1024, activation_fn=tf.nn.relu)
    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, name='dropout_layer')
    
    print('\nVierte Schicht: Vollständig verknüpft -lineare aktivierung-')
    h4 = fc_layer(h2_pool, name= 'fc_4', n_output_units=10, activation_fn=None)
    
    # Vorhersage
    predictions = {'probabilities': tf.nn.softmax(h4, name='probabilities'),
                   'labels': tf.cast(tf.argmax(h4, axis=1), dtype=tf.int32, name='labels')}
    
    
    ## Visualisierung des Graphen mit TensorBoard
    ## Verlustfunktion und Optimierung
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4, labels=tf_y_onehot),
                                        name='cross_entropy_loss')
    
    # Optimierung:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')
    
    # Berechnung der Korrektklassifizierungsrate
    correct_predictions = tf.equal(predictions['labels'], tf_y, name='correct_preds')
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accurycy')
    
if __name__=='__main__':
    
    g = tf.Graph()
    with g.as_default():
        y = tf.constant(np.random.rand(1,28,28,3), dtype=tf.float32)
        x = tf.placeholder(dtype=tf.float32, shape=[None, 28,28,1]) #[batch, width, height, channels (RGB or Grey)]
        conv_layer(y, name='convtest', kernel_size=(3,3), n_output_channels=32, padding_mode='SAME', strides=(1,1,1,1))
        fc_layer(y, name='fctest', n_output_units=32, activation_fn=tf.nn.relu)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            a = [i for i in tf.global_variables()][0]
            b = a.eval()
    
    del g,x




























