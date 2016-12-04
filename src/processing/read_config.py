# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 13:02:23 2016

@author: Bill Eggert
"""

import abc
from loader_base import loader_base
from Tkinter import Tk
from tkFileDialog import askopenfilename

class load_config(loader_base):
    
    def get_path(self,path)
    Tk().withdraw() 
    filename = askopenfilename()
    print(filename)
    
    