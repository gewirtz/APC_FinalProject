# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 15:42:10 2016

@author: Bill Eggert
"""


class loader_base(object):
    
    @abc.abstractmethod
    def load_data(self):
        return
        
    @abc.abstractmethod
    def save_data(self,output,data):
        """ save data object to output"""
        return
    