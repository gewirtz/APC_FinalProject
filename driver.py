import os
import sys
import json

#file = sys.argv[1]
file = 'metadata_configurations.json'

#Read the metadata
metadata = json.load(open(file))


# Pre-Processing
ncores_prep = metadata['pre_processing']['ncores']
if ncores_prep == 1: 
	# Serial Implementation of pre-processing
	os.system('python driver_pre_processing.py %s 1' % file)
else:
	# Parallel Implementation of model fitting
	os.system('mpirun -np %i python driver_pre_processing.py %s %i' % (ncores_prep,file,ncores))


# Model Fitting
ncores_modf = metadata['model_fitting']['ncores']
if ncores_modf == 1: 
	# Serial Implementation of model fitting
	os.system('python driver_model_fitting.py %s 1' % file)
else:
	# Parallel Implementation of model fitting
	os.system('mpirun -np %i python driver_model_fitting.py %s %i' % (ncores_modf,file,ncores_modf))


# Graphical Pos-Processing
# to be implemented