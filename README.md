# APC_FinalProject

# 1. Compile C++ Modules
make

# 2. Edit Configuration File
vim config/config.json

#  2.1 Set your training and testsign directories in:
#     "traning_dir" and "testing_dir"
#
#  2.2 Set your image file and label files:
#     "train_img" and "train_lbl" for training and 
#     "test_img" and "test_lbl" for testing
#
#  2.3 Set processing modules in "processing" with the following options":
#      processing: 0 for no processing
#      processing: 1 for histogramming
#      processing: 2 for gaussian smoothing
#
#  2.4 Set model fitting modlues in "model_fitting" with the following options:
#      model_fitting: 0 for Linear Regression
#      model_fitting: 1 for Regularized Regression
#      model_fitting: 2 for Logistic Regression
#      model_fitting: 3 for kNN
#      model_fitting: 4 for Naive Bayes
# 
#  2.5 Set the "test_unit" variable, 0 for normal run, 1 for testing

# 3. Run Simulation
python ./driver.py

