# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 13:08:17 2016

@author: Bill Eggert
"""
""" I couldn't get the abstract class sorted out in one day,
so here's a single pass thru for data processing
"""

# "deskewing processing"

import os, struct
import numpy as np
from array import array
# step 1: import mmnist data