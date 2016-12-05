import os
import sys
import json


home_path=os.getcwd()
config_file = '%s/config/metadata_configurations.json' % home_path

#Read the metadata
metadata = json.load(open(config_file))

# Image Processing
ncores = metadata['pre_processing']['ncores']
if ncores == 1: 
	# Serial Implementation of pre-processing
	os.system('python %s/src/processing/driver_pre_processing.py %s %s 1' % (home_path,home_path,config_file))
else:
	# Parallel Implementation of model fitting
	os.system('mpirun -np %i python %s/src/processing/driver_pre_processing.py %s %s %i' % (ncores,home_path,config_file,ncores))


# Model Fitting
methods_fitting = metadata['model_fitting']['methods_to_simulate']
for met in methods_fitting:
	print met
	#os.system('%s/src/model_fitting/ARI_AND_CHASE_CODE ./config/%s ' % (home,file))


# Graphical Pos-Processing
# to be implemented