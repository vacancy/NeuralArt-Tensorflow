# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# 
# This file is part of LibNeuralArt 

import tensorflow as tf

from .base import ModelBase, NeuArtist2Loader

__all__ = ['VGG16']


class VGG16(ModelBase):
    def __init__(self, weight_path, inp_h=None, inp_w=None, name='VGG16'):
        super().__init__(name=name)

        self._loader = NeuArtist2Loader(weight_path)
        self._inp_h = inp_h
        self._inp_w = inp_w

    def _initialize():
        def build(inpvar, layer_name):
            if layer_name.startswith('conv'):
                W, b = self._loader.get_conv_weight(layer_name)
                out = tf.nn.relu(tf.nn.conv2d(inpvar, W, strides=[1, 1, 1, 1], padding='SAME') + b)
            else:
                assert layer_name.startswith('pool')
                out = tf.nn.max_pool(inpvar, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self._add_var(layer_name, out)
            return out

        _ = tf.Variable(np.zeros((1, self._inp_h, self._inp_w, 3)).astype('float32'))
        self._add_var('input', _)
        _ = _ - np.array([123, 117, 104]).reshape((1, 1, 1, 3))
        self._add_var('normed_input', _)

        for i, nr_layers in enumerate([2, 2, 3, 3, 3]):
            for j in range(nr_layers):
                _ = build(_, 'conv{}_{}'.format(i+1, j+1))
            _ = build(_, 'pool{}'.format(i+1))

