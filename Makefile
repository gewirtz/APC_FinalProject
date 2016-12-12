driver = alpha_driver.o

#ARMA_INCLUDE_FLAG = -I ../include
LIB_FLAGS = -larmadillo
OPT = -O2
CXX = g++

CXXFLAGS = $(OPT)

all: alpha_driver

mnist_load : $(driver)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB_FLAGS)

clean:
	$(RM) *.o
	$(RM) .depend

depend:
	$(CXX) -MM $(CXXFLAGS) *.cc > .depend

-include .depend
