# Noemi Vergopolan
# 

import os
import sys
import simplejson

home_path=os.getcwd()
config_file = '%s/config/config.json' % home_path

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
t_unit = metadata["test_unit"]

cmd = './main "%s" "%s" "%s" "%s" "%s" "%s" %i %i %i' % (train_dir, test_dir, train_lbl, train_img, test_lbl, test_img)
print cmd
os.system(cmd)



