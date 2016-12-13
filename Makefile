MNIST_DEPS = processing/mnist_load_images.cc processing/mnist_load_labels.cc processing/mnist_count_images.cc

driver = $(MNIST_DEPS) alpha_driver.cc

#ARMA_INCLUDE_FLAG = -I ../include
LIB_FLAGS = -larmadillo
OPT = -O2
CXX = g++

CXXFLAGS = $(OPT)

all: alpha_driver

#alpha_driver : $(driver)
#	$(CXX) $(CXXFLAGS) $@ $^ $(LIB_FLAGS)


# I hate make files
alpha_driver : $(driver)

	g++ alpha_driver.cc processing/mnist_load_images.cc processing/mnist_load_labels.cc processing/mnist_count_images.cc processing/no_processing_test.cc processing/no_processing.cc -o alpha_driver

clean:
	$(RM) *.o
	$(RM) .depend

depend:
	$(CXX) -MM $(CXXFLAGS) *.cc > .depend

-include .depend
