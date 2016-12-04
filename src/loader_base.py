# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 15:42:10 2016

@author: Bill Eggert
"""
from abc import ABCMeta, abstractmethod

class BaseLoader(metaclass = ABCMeta):
    
    @abc.abstractmethod
    def get_data(self):
        print("Hello World")
    