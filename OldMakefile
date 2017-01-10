
MNIST_DEPS = processing/mnist_load_images.cc processing/mnist_load_labels.cc processing/mnist_count_images.cc
PROCESS_DEPS = processing/no_processing.cc processing/no_processing_test.cc
MF_DEPS = ModelFitting/GradientDescent.cpp ModelFitting/LinearRegression.cpp

driver = $(MNIST_DEPS) $(PROCESS_DEPS) $(MF_DEPS) alpha_driver.cc

#ARMA_INCLUDE_FLAG = -I ../include
LIB_FLAGS = -larmadillo 
OPT = -O2
CXX = g++ -std=c++11
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


