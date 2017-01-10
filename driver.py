# Noemi Vergopolan
# 

import os
import sys
import json
import simplejson

home_path=os.getcwd()
config_file = '%s/config/config.json' % home_path
print config_file

#Read the metadata
metadata = simplejson.load(open(config_file))

preprocessing = metadata['preprocessing']
model_fitting = metadata['model_fitting']
train_dir = home_path+metadata["dir"]["traning_dir"]
test_dir = home_path+metadata["dir"]["testing_dir"]
train_lbl = metadata["files"]["train_lbl"]
train_img = metadata["files"]["train_img"]
test_lbl = metadata["files"]["test_lbl"]
test_img = metadata["files"]["test_img"]


for i in range(len(preprocessing)):
	preprocess_met = preprocessing[i]
	model_fit_met = model_fitting[i]
        cmd = './alpha_drive %s %s %s %s %s %s %s %s' % (train_dir, test_dir, train_lbl, train_img, test_lbl, test_img, preprocess_met, model_fit_met)
	print cmd
	#os.system(cmd)



# Graphical Pos-Processing
# to be implemented
