
# coding: utf-8

# In[25]:

get_ipython().magic(u'load_ext Cython')
%%cython --annotate
from data_import_first_try.py import images
from data_import_first_try.py import labels
from libc.stdlib cimport malloc
import numpy as np
cimport numpy as np
#Simple array return
cdef public np.ndarray getImages():
    return <np.ndarray>images
cdef public np.ndarray getLabels():
    return <np.ndarray>labels
#Array shape
cdef public int getShape(np.ndarray src):
    cdef int size[3]
    for i in range(3):
        size[i]=src.shape[i]
    return <int>size

#Copy data from numpy array 
cdef public void copyData(float *** array,np.ndarray src):
    cdef float ***tmp
    cdef int i, j, k, m = src.shape[0], n=src.shape[1], nn=src.shape[2]; 
    # Allocate initial pointer 
    tmp=<float ***>malloc(m*sizeof(float **))
    if not tmp:
        raise MemoryError()
    #Allocate planes
    for i in range(m):
        tmp[i]=<float**>malloc(n*sizeof(float*))
    #Allocate elements
    for j in range(n):
        tmp[i][j]=<float*>malloc(nn*sizeof(float))
    # Copy numpy Array
    for i in range(m):
        for j in range(n):
            for k in range(nn):
                tmp[i][j][k] = src[i,j,k]
    # Assign pointer to dst
    array = tmp

