import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

g = tf.Graph()

with g.as_default():
    
    t1 = tf.constant([np.pi], dtype=tf.float32, verify_shape=False, name='t1')
    t2 = tf.constant(np.random.rand(10), name='t2')
    t3 = tf.constant(np.random.rand(3,3), name='t3')
    
    
    
    
    
    t1_rank = tf.rank(t1, name='t1_rank')
    
    
    t1_shape = t1.get_shape()
    
    print(t1_shape)
    print(t2.as_list())
    print(t3)
    with tf.Session() as sess:
        print(sess.run([t3]))
        
        
del g