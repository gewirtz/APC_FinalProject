# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 13:08:17 2016

@author: Bill Eggert
"""
""" I couldn't get the abstract class sorted out in one day,
so here's a single pass thru for data processing
"""

# adapted from "traditional" python script for reading MNISt data"

import os, struct
import numpy as np
from array import array as pyarray
import matplotlib.pyplot as plt
# step 1: import mmnist data

# goal: load mnist data into 3D numpy arrays
def load_mnist(dataset, digit, path):
    
    
    
    if dataset == "training":
        image_path = path + '/train-images.idx3-ubyte'
        label_path = path + '/train-labels.idx1-ubyte'
    elif dataset == "testing":
        image_path = os.path.join(path, 't10k-images-idx3-ubyte')
        label_path = os.path.join(path, 't10k-labels-idx1-ubyte')
        
    with open(label_path, 'rb') as file:
        num, size = struct.unpack(">II", file.read(8))
        # error checking
        label_data = pyarray("b",file.read())

    with open(image_path,'r') as file:
        num, size, rows, cols = struct.unpack(">IIII", file.read(16))    
        image_data = pyarray("B",file.read())
      
    print(label_data[0])
    ind = [i for i in range(size) if label_data(i) in digit]
    N = len(ind)
    
    for i in range(N):
        images[i] = np.array(image_data[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
        images[i].reshape((rows,cols))
        
        labels[i] = label_data[ind[i]]
        
    return images, labels

test_path = "C:/Users/billi/Desktop/GitHub Repo/APC_FinalProject/data/mnist/testing"
train_path = "C:/Users/billi/Desktop/GitHub Repo/APC_FinalProject/data/mnist/training"

images, labels = load_mnist(dataset = "training", digit = [5], path = train_path)
imshow(images.mean(axis=0), cmap = cm.gray)
show()