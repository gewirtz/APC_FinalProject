# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 15:42:10 2016

@author: Bill Eggert
"""

import abc
from cStringIO import StringIO

class loader_base(object):
    __metaclass__ = abc.ABCMeta
    

    @abc.abstractmethod
    def conf(self, path = '.'):
        """retrieve config data and return object"""
        return

    
    @abc.abstractmethod
    def load_data(self,input):
        print('base class reading data')
        """retrieve data and return an object"""
        return input.read()
        
    @abc.abstractmethod
    def save_data(self,output,data):
        """ save data object to output"""
        return
    