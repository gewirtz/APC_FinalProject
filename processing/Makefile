mnist = mnist_load.o

#ARMA_INCLUDE_FLAG = -I ../include
#LIB_FLAGS = -lblas -llapack -larmadillo -lm
# OPT = -O2
CXX = g++

# CXXFLAGS = $(ARMA_INCLUDE_FLAG) $(OPT)

all: mnist_load

#mnist_load : $(mnist)
#	$(CXX) $(CXXFLAGS) -c  $@ $^ $(LIB_FLAGS)

mnist_load : $(mnist)
	$(CXX) -o $@ $^

clean:
	$(RM) *.o
	$(RM) .depend

depend:
	$(CXX) -MM $(CXXFLAGS) *.cc > .depend

-include .depend