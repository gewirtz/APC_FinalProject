import os
import sys
import json

#file = sys.argv[1]
file = 'metadata_configurations.json'

#Read the metadata
metadata = json.load(open('./config/%s' % file))
home=metadata['data_input']['home_folder']


# Image Processing
ncores = metadata['pre_processing']['ncores']
if ncores == 1: 
	# Serial Implementation of pre-processing
	os.system('python %s/src/processing/driver_pre_processing.py ./config/%s 1' % (home,file))
else:
	# Parallel Implementation of model fitting
	os.system('mpirun -np %i python %s/src/processing/driver_pre_processing.py ./config/%s %i' % (ncores,home,file,ncores))


# Model Fitting
methods_fitting = metadata['model_fitting']['methods_to_simulate']
for met in methods_fitting:
	print met
	#os.system('%s/src/model_fitting/ARI_AND_CHASE_CODE ./config/%s ' % (home,file))


# Graphical Pos-Processing
# to be implemented