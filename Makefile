MNIST_DEPS = processing/mnist_load_images.cc processing/mnist_load_labels.cc processing/mnist_count_images.cc processing/ppm_load_images.cc
# processing/jpg_load_images.cc 


PROCESS_DEPS = processing/no_processing.cc processing/no_processing_test.cc processing/gaussian_smoothing.cc processing/gs_processing_test.cc processing/histogram.cc processing/histogram_test.cc


MF_DEPS = ModelFitting/GradientDescent.cpp ModelFitting/LinearRegression.cpp ModelFitting/LogisticRegression.cpp ModelFitting/KNN.cpp ModelFitting/CrossValidation.cpp Performance/Performance.cpp 

driver = $(MNIST_DEPS) $(PROCESS_DEPS) $(MF_DEPS) main.cpp

#ARMA_INCLUDE_FLAG = -I include
LIB_FLAGS = -larmadillo -lpython2.7
OPT = -O2
CXX = g++ -std=c++11
CXXFLAGS = $(OPT) 

all: main

demo_driver : $(driver) 
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB_FLAGS)

clean:
	$(RM) *.o
	$(RM) .depend

depend:
	$(CXX) -MM $(CXXFLAGS) *.cc > .depend

-include .depend


