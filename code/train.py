#!/usr/bin/env python

import os
import time
from model_patient import *
from data_helper import load_data
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 256


trX, trY, deX, deY, teX, teY = load_data()
word_embedding = np.load('../data/model_50.npy')
py_x,cost  = model()

train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = py_x

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
        assert y.shape[0] == py.shape[0]
        return np.mean( y * py > 0)

    best_dev_acc = -1
    best_test_acc = -1
    saver = tf.train.Saver()
    
    model_dir = '../result/model/{:s}'.format(time.strftime("%Y-%m-%d", time.localtime()))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    for i in range(10000):
        train_indices = range(len(trX))
        np.random.shuffle(train_indices)
        trX = trX[train_indices]
        trY = trY[train_indices]

        training_batch = zip(range(0, len(trX) - batch_size, batch_size),
                             range(batch_size, len(trX)+1 - batch_size, batch_size))
        loss_list = []
        for start, end in training_batch:
            y = trY[start:end] == trY[start+batch_size:end+batch_size]
            _,loss = sess.run([train_op, cost], feed_dict={
                X1: word_embedding[trX[start:end]],
                X2: word_embedding[trX[start+batch_size:end+batch_size]],
                Y: 2 * (y -0.5),
                p_keep_hidden: 0.5
                })
            loss_list.append(loss)
        loss = np.mean(loss_list)

        dev_acc = test(deX,deY)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            test_acc = test(teX,teY)
            best_test_acc = max(test_acc, best_test_acc)
            saver.save(sess,os.path.join(model_dir, 'dev-{:d}--test-{:d}-model'.format(int(dev_acc*100),int(test_acc*100))))
            saver.save(sess,('../result/model/best/best'))
        print(i,'dev_acc:',dev_acc, 'test_acc:', test_acc, 'best_dev_acc:',best_dev_acc, 'best_test_acc:', best_test_acc, 'loss:',loss)
