import os
import sys
import json

#file = sys.argv[1]
file = '/tigress/nvrocha/projects/APC524/config/metadata_configurations.json'

#Read the metadata
metadata = json.load(open(file))


# Pre-Processing
ncores_prep = metadata['pre_processing']['ncores']
if ncores_prep == 1: 
	# Serial Implementation of pre-processing
	os.system('python ./src/processing/driver_pre_processing.py %s 1' % file)
else:
	# Parallel Implementation of model fitting
	os.system('mpirun -np %i python ./src/processing/driver_pre_processing.py %s %i' % (ncores_prep,file,ncores))


# Model Fitting
# Serial Implementation of model fitting
#os.system('python ./src/model_fitting/driver_model_fitting.py %s 1' % file)


# Graphical Pos-Processing
# to be implemented