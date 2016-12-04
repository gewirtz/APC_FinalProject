# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 15:16:30 2016

@author: Bill Eggert

Hello World to APC524
"""

class mnist(BaseLoader):
    def get_data(self, someOption = False):
        a = A() if someOption else None
        return self.what + a
        