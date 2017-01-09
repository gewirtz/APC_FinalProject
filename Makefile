MNIST_DEPS = processing/mnist_load_images.cc processing/mnist_load_labels.cc processing/mnist_count_images.cc
PROCESS_DEPS = processing/no_processing.cc processing/no_processing_test.cc processing/gaussian_smoothing.cc processing/gs_processing_test.cc processing/histogram.cc processing/histogram_test.cc
MF_DEPS = ModelFitting/GradientDescent.cpp ModelFitting/LinearRegression.cpp

driver = $(MNIST_DEPS) $(PROCESS_DEPS) alpha_driver.cc

#ARMA_INCLUDE_FLAG = -I ../include
LIB_FLAGS = -larmadillo -llapack -lblas -lgfortran
OPT = -O2
CXX = g++
CXXFLAGS = $(OPT)

all: alpha_driver

alpha_driver : $(driver)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB_FLAGS)

clean:
	$(RM) *.o
	$(RM) .depend

depend:
	$(CXX) -MM $(CXXFLAGS) *.cc > .depend

-include .depend
