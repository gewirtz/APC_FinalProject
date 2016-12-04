# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 15:42:10 2016

@author: Bill Eggert
"""


class loader_base(object):
    
    def __init__(self):
        self.test_path "C:\Users\billi\Desktop\GitHub Repo\APC_FinalProject\data\mnist\testing"
        self.train_path = "C:\Users\billi\Desktop\GitHub Repo\APC_FinalProject\data\mnist\training"
        
        self.train_images = []
        self.test_images = []

    def load_data(self):
        return
        
    @abc.abstractmethod
    def save_data(self,output,data):
        """ save data object to output"""
        return
    