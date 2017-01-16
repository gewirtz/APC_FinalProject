# APC_FinalProject

# How to Run:
# 1. Compile C++ Modules
make

# 2. Edit Configuration File
vim config/config.json

#  2.1 Set your training and testsign directories in:
     "traning_dir" and "testing_dir"

#  2.2 Set your image file and label files:
     "train_img" and "train_lbl" for training and 
     "test_img" and "test_lbl" for testing

# 3. Execute
 There is two ways you can execute the file. The first is through a python driver, which will read input data directories from confi.json:
 
python ./driver.py

Alternatively, you can execute and pass input file folders by hand using:

./main train_dir test_dir train_lbl train_img test_lbl test_img

# 4. Model Options
The user will be able to set additional configurations parameters through i/o communication with the user via terminal:

#  4.1 Set processing modules in "processing" with the following options":
      processing: 0 for no processing
      processing: 1 for histogramming
      processing: 2 for gaussian smoothing

#  4.2 Set model fitting modlues in "model_fitting" with the following options:
      model_fitting: 0 for Linear Regression
      model_fitting: 1 for Regularized Regression
      model_fitting: 2 for Logistic Regression
      model_fitting: 3 for kNN


# 5. Data Processing:

data_process_base.h:
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


# 6. Model Fitting:
# 6.1 Data Processing:

model.h: pure virtual class for Model objects.
Variables: 
Functions: Destructor
			predict - will take a matrix of examples and return a vector of predicted labels for them
			set_Params: will set the kth entry in the parameter vector to the vector p
			get_Params: will take nothing and return the vector of vectors where the parameters for the model are stored
			get_num_examples: will take nothing and return the number of training examples provided to the model
			getTrainset: will take nothing and return the matrix of training examples
			getLabels: will take nothing and retrn the vector of training labels.
			getLabelSet: will take nothing and return the set of all possible labels.


KNN.h/.cpp:
Variables: trained: Flag to set whether the model has been trained yet, to make sure that normalization applies to the testing set the same as the training set.
			normalize: Flag to set whether the data should be normalized (a flag set in the main portion of the program)
			optim: the optimizer object associated with the model.
			num_examples: the number of training examples
			x: the matrix of training data
			y: the vector of training classification labels:
			params: For KNN, this is just the k chosen by the optimizer used to choose k nearest neighbors.
			label_set: a set of all possible labels for classification
			tr_means: row vector of means from the training examples
			tr_stdev: row vector of standard deviations from the training examples
			remove: vector of booleans representing whether columns in x are collinear or uninformative and should be removed
			num_regressors: number of informative columns, used to allocate matrix size
			initial_regressors: number of total initial columns (features) before trimming
Functions: Constructor: takes in training examples, training labels, an Optimizer object, and a flag saying whether the data should be normalized. It sets several member variables, normalizes the data if the flag is true, shuffles the examples in case the user has provided them in order, and calls the fit method. Doesn't return anything.
	Fit: takes no arguments, calls the Optimizer's method fitParams, passing itself as an object.
	Predict: takes a matrix of test examples as an argument. If the normalize flag is true, it normalizes the testing data. It calculates the distance matrix for each test example to each train example. It calls internal_predict and returns the output from that function. Predict is called from the main program.
	InternalPredict: takes a matrix of test examples, a matrix of training examples, k, a vector of the labels for the training examples, and a distance matrix of the L2 norm between each testing and training example. It finds the mode of the labels of the k closest neighbors to each test example from the training set and sets that as the predicted label for that test example.  This returns a vector of the predicted labels for the test examples. This is a private function.
	Predict_on_subset: Takes a matrix of test samples, a matrix of training examples, k, a vector of the labels for the training examples, and a distance matrix of the L2 norm between each testing and training example. It passes these arguments straight into internal_predict and returns the output from that function. Intended to be called from the CrossValidation Optimizer object-- ie predict labels on a subset of the training data.
	set_Params: a public method to set k, called by the Optimizer object to set the best k it found as a member variable for the KNN object. It takes an integer k , which is NOT the k to set--this is just the index in the parameter vector to set equal to the second parameter-- a vector which here is just a vector containing the best k for KNN, as found by CrossValidation. This doesn't return anything.
	get_Params: takes nothing, returns the vector of parameters. The first element here is a vector, the first element is the k selected by cross validation.
	getTrainset: public method which takes no parameters and returns x.
	getLabels: public method which takes no parameters and returns y.
	get_num_examples: public method which takes no parameters and returns num_examples
	getLabelSet: public method which takes no parameters and returns label_set.
	standardize: private method to standardize a matrix of examples.

# 6.2 Optimizers:

Optimizer.h: Abstract class for Optimizer objects.
Variables: None
Functions: Destructor
			fitParams- takes a model object. In the specific implementations of Optimizer objects this function fits the parameters in the model that it is associated to.  

CrossValidation.h/.cpp:
Conducts cross validation to select the best hyperparameter from a given interval.
Variables: param_range_start: the start of the possible values for the hyperparameter
			param_range_end: the upper limit of the possible hyperparameter values
			delta: the step size to take through the possible parameter value range
			nfolds: number of folds to partition the data set into (ie nfolds-cross validation)
Functions: Constructor: takes in param_range_start, param_range_end, delta, and nfolds. Uses these to set its member variables.
			Destructor: empty
			fitParams: takes in a model object and uses its training data to calculate
			the best value for the hyperparameter in question. For KNN, it calls the function calculate_dists. fitParams then calls a setter from the model to store the best hyperparameter as a member variable for the specific model.
			calculate_dists: takes in a matrix of training data and returns a matrix of distances from each example to every other example. This is specifically used for KNN.




