# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# 
# This file is part of LibNeuralArt 

import pickle
import numpy as np

from nart.logconf import logger

class ModelBase(object):
    def __init__(self, name=None):
        self._name = name
        self._vars = dict()
        self.__initialized = False

    @property
    def name(self):
        return self._name

    def initialize(self):
        if self.__initialized:
            return 
        self._initialize()
        self.__initialized = True

    def _initialize(self):
        raise NotImplementedError('model must override _initialize method')

    def __getitem__(self, name):
        self.initialize()
        if name in self._vars:
            return self._vars[name]
        return None

    def _add_var(self, name, var):
        if name in self._vars:
            logger.warn('duplicated adding var: {}'.format(name))
        self._vars[name] = var


class NeuArtist2Loader(object):
    ''' proxy for NeuArtist2 weights '''
    def __init__(self, path):
        self.__path = path
        self.__storage = None

    def load_weights(self):
        if self.__storage is not None:
            return 
        
        logger.info('initialize weight from {}'.format(self.__path))
        with open(self.__path, 'rb') as f:
            self.__storage = pickle.load(f)
    
    def get_conv_weight(self, layer_name):
        self.load_weights()

        weight = self.__storage['params'][layer_name + ':W']
        weight = weight.transpose([2, 3, 1, 0])
        bias = self.__storage['params'][layer_name + ':b']
        return weight, bias

    def get_fc_weight(self, layer_name):
        self.load_weights()

        weight = self.__storage['params'][layer_name]
        bias = self.__storage['params'][layer_name + ':b']
        return weight, bias

