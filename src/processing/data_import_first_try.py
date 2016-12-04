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
from numpy import append, array, int8, uint8, zeros
from array import array as pyarray
import matplotlib as mpl
from matplotlib import pyplot
# step 1: import mmnist data

# goal: load mnist data into 3D numpy arrays
def load_mnist(dataset, digit, path):
    
    """
    Returns a 3D array
    each image[i] is a 2D array of the handwriting image
    
    """
    
    if dataset == "training":
        image_path = path + '/train-images.idx3-ubyte'
        label_path = path + '/train-labels.idx1-ubyte'
    elif dataset == "testing":
        image_path = path + '/t10k-images-idx3-ubyte'
        label_path = path + '/t10k-labels-idx1-ubyte'
    
    with open(label_path,'rb') as file:
        num, size = struct.unpack(">II", file.read(8))
        label_data = pyarray("b",file.read())
        
    with open(image_path,'rb') as file:
        num, size, rows, cols = struct.unpack(">IIII", file.read(16))
        image_data = pyarray("B", file.read())
        
    ind = [i for i in range(size) if label_data[i] in digit]
    N = len(ind)
    
    #preallocate
    images = zeros((N,rows,cols), dtype=uint8)
    labels = zeros((N,1), dtype=int8)
    
    
    for i in range(N):
        
        images[i] = array(image_data[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = label_data[ind[i]]
        
    return images, labels
        

test_path = "C:/Users/billi/Desktop/GitHub Repo/APC_FinalProject/data/mnist/testing"
train_path = "C:/Users/billi/Desktop/GitHub Repo/APC_FinalProject/data/mnist/training"
testdigit = [2, 5, 8]
images, labels = load_mnist(dataset = "training", digit = testdigit, path = train_path)


testnum = 2
# let's see if it works
fig = pyplot.figure()
ax = fig.add_subplot(1,1,1)
image_plot = ax.imshow(images[testnum],cmap = mpl.cm.gray)
print(labels[testnum])