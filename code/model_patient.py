#!/usr/bin/env python

import tensorflow as tf


num_of_words = 250
dimension_of_word_embeddings = 50
# num_of_words = 28
# dimension_of_word_embeddings = 28
num_of_filters = 100
X1 = tf.placeholder("float", [None, num_of_words, dimension_of_word_embeddings])
X2 = tf.placeholder("float", [None, num_of_words, dimension_of_word_embeddings])
Y = tf.placeholder("float", [None])
p_keep_hidden = tf.placeholder("float")


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model():


    def get_feature(X):
        X = tf.reshape(X,[ -1, num_of_words, dimension_of_word_embeddings,1])
        w = init_weights([3,dimension_of_word_embeddings,1,num_of_filters])
        conv = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='VALID'))
        conv = tf.nn.max_pool(conv, ksize=[1, num_of_words, 1, 1], strides=[1, num_of_words, 1, 1], padding='SAME')
        conv = tf.reshape(conv, [-1, conv.get_shape().as_list()[-1]])
        return conv

    conv1 = get_feature(X1)
    conv2 = get_feature(X2)

    similarity_matrix = init_weights([num_of_filters, num_of_filters])
    # similarity_matrix = similarity_matrix + tf.transpose(similarity_matrix)


    conv1 = tf.matmul(conv1, similarity_matrix)
    similarity = tf.reduce_sum(conv1 * conv2, axis=1)
    similarity = tf.reshape(similarity, [-1, 1])

    # print similarity.get_shape()
    pairwise_feature = tf.concat(1,[conv1,conv2,similarity])
    pairwise_feature = tf.nn.dropout(pairwise_feature, p_keep_hidden)
    # print pairwise_feature.get_shape()

    w = init_weights([num_of_filters * 2 +1,1])
    y = tf.matmul(pairwise_feature,w)
    y = tf.reshape(y, [-1])

    cost = tf.reduce_mean((Y - y) ** 2)
    # print y.get_shape()
    # print cost.get_shape()
    return y, cost


# model()

