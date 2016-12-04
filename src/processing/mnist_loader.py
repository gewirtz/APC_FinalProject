# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 15:16:30 2016

@author: Bill Eggert

"""

import abc
from loader_base import loader_base
from cStringIO import StringIO


class mnist_loader(loader_base):
    
    def load_data(self,input):
        base_data = super(mnist_loader,self).load_data(input)
        print('subclass sorting data')
        response = sorted(base_data.splitlines())
        return response
        
    def save_data(self,output,data):
        return output.write(data)

        ""
mnist_loader.load_config        
input = StringIO(""" line one
line two
line three
""")
reader = mnist_loader()
print reader.load_data(input)
print
"""
if __name__ == '__main__':
    print('Subclass:',issubclass(mnist_loader, loader_base))
    print('Instance:',isinstance(mnist_loader(), loader_base))
 """   
    
    

        