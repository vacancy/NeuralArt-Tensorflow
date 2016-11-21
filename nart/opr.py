# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# 
# This file is part of LibNeuralArt 


import tensorflow as tf


def conv2d(inpvar, W, b, nonlin):
    _ = inpvar
    _ = tf.nn.conv2d(_, tf.Variable(W), strides=(1, 1, 1, 1), padding='SAME')
    _ = tf.nn.bias_add(_, b)
    _ = nonlin(_)
    return _


def pooling2d(inpvar, method='MAX'):
    assert method == 'MAX'
    return tf.nn.max_pool(inpvar, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

