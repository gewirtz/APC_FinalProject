# APC_FinalProject

How to Run:
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
# 
#  2.5 Set the "test_unit" variable, 0 for normal run, 1 for testing

# 3. Run Simulation //WE ARE NOT RUNNING SIMULATIONS 
python ./driver.py //IS THIS ACTUALLY HOW'S IT'S RUN BC I DON'T THINK IT IS? - NINA

Data Processing:
data_process__base.h:
Virtual class for all derived data processing types. 
Variables: train data and labels, test data and labels, flag for train having been processed, flag for test having been processed
Functions: accesssors and modifiers for about variables (not including flags)
Process function - Has no inputs but runs off of internal flags to avoid user interaction (i.e. user processing wrong data or data twice). Running once processes train, the second time processes test, all other runs after that do nothing. The exact mechanics of the function depend on the processing type. Processed data overwrites the data members - no_processing will always keep the raw data.

no_processing.h/.cpp:
The process function here returns the raw data.

gaussian_smoothing.h/.cpp:
Uses seperability of Gaussian kernel to do horizontal/vertical convolutions seperately. Implements a sliding window approach to avoid needing additional data storage. Kernel is 5x5. Mirroring is used for border conditions. Process function does not change image or dataset dimensions - just the pixel values contained therein. 

histogram.h/.cpp:
Frequency counts of pixels in each image. Resulting histogram is matrix of n image X k possible pixel values (0-255 since we are using grayscale). The histogram is added to the front of the dataset being processed - thereby adding one image to the vector length. This is changed in the driver function for compatibility with the modeling portion. 


