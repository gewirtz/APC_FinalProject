import warnings
warnings.filterwarnings('ignore')
import cPickle as pickle
import sys
import os
import numpy as np
import json
import glob


#Get general info
home_path = sys.argv[1]
config_file = sys.argv[2]
ncores = sys.argv[3]

if ncores == 1: 
	size = 1
	rank = 0
else:
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()

#Read in the metadata
metadata = json.load(open(config_file))

methods = metadata['pre_processing']['methods_to_simulate']

for mets in methods:
	pass

os.system('python %s/src/processing/dev/data_import_first_try_Noemi.py %s %s' % (home_path,home_path,config_file))


# #List of pre processing methods to be used
# methods_prep = metadata['model_fitting']['methods_to_simulate']

# #Work on each image

# for i_image in np.arange(n_traning)[rank::size]:

# 	#print rank, n_traning, np.arange(n_traning)[rank::size]
# 	#exit()
	
# 	print "Rank:%d, Image:%d - Initializing" % (rank,i_image)

# 	#for met in methods_prep:
# 		#Pass images, methods, and parameters to Pre Processing module
# 		#os.system('python ./Pre_Processing.py %s %s %s' % (i_image,met,metadata_file))
# 		# Or receveid and save in different files


