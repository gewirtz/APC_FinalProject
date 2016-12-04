import warnings
warnings.filterwarnings('ignore')
import cPickle as pickle
import sys
import os
import numpy as np
import json
import glob


#Get general info
metadata_file = sys.argv[1]
ncores = sys.argv[2]

if ncores == 1: 
	size = 1
	rank = 0
else:
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()

#Read in the metadata
metadata = json.load(open(metadata_file))

#Retrive a list of all the traning and testing files
traning_files=glob.glob('%s/*' % metadata['data_input']['traning_folder'])
testing_files=glob.glob('%s/*' % metadata['data_input']['testing_folder'])

#Determine the number of training images
n_traning = len(traning_files)
n_testing = len(testing_files)

#List of pre processing methods to be used
methods_prep = metadata['model_fitting']['methods_to_simulate']

#Work on each image

for i_image in np.arange(n_traning)[rank::size]:

	print rank, ncatch, np.arange(ncatch)[rank::size]
	exit()
	#Print info
	print "Rank:%d, Image:%d - Initializing" % (rank,i_image)

	for met in methods_prep:
		#Pass images, methods, and parameters to Pre Processing module
		#os.system('python ./Pre_Processing.py %s %s %s' % (i_image,met,metadata_file))
		# Or receveid and save in different files



