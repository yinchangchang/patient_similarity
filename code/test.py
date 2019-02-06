#!/usr/bin/env python

import os
import time
from model_patient import *
from data_helper import load_data, load_data_all
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 256


allX, allY = load_data_all()
word_embedding = np.load('../data/model_50.npy')



# print teY
# raw_input()
py_x,cost  = model()

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = py_x

# print trY.shape


# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    def test(teX,teY):
        test_indices = range(len(teX)) # Get A Test Batch
        test_indices_1 = test_indices[:]
        np.random.shuffle(test_indices)
        test_indices_2 = test_indices[:]
        y = teY[test_indices_1] == teY[test_indices_2]
        y = 2 * (y - 0.5)
        py = sess.run(predict_op, feed_dict={
                             X1: word_embedding[teX[test_indices_1]],
                             X2: word_embedding[teX[test_indices_2]],
                             p_keep_hidden: 1.0})
        if y.shape[0] != py.shape[0]:
            print err
        return np.mean( y * py > 0)

    saver = tf.train.Saver()
    
    # model_dir = '../result/model/{:s}'.format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    # model_dir = '../result/model/{:s}'.format(time.strftime("%Y-%m-%d", time.localtime()))
    model_dir = '../result/model/best'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    saver_new = tf.train.import_meta_graph(model_dir+ '/best.meta')
    saver_new.restore(sess, tf.train.latest_checkpoint(model_dir))
    test_acc = test(allX,allY)
    print test_acc
