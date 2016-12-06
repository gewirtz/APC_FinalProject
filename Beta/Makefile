optimizer = GradientDescent.o
method =   LinearRegression.o 
#objects1 =   $(integrators) $(equations1) ode_solve.o
#objects2 =   $(integrators) $(equations2) test_convergence.o

ARMA_INCLUDE_FLAG = -I ../include
LIB_FLAGS = -lblas -llapack 
OPT = -O2
CXX = g++
CXXFLAGS = -Wall $(ARMA_INCLUDE_FLAG) $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT)

all: LinearRegression GradientDescent

GradientDescent : $(optimizer)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB_FLAGS)

LinearRegression : $(method)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB_FLAGS)

clean:
	$(RM) *.o
	$(RM) .depend

depend:
	$(CXX) -MM $(CXXFLAGS) *.cpp > .depend

-include .depend


#CXX=CC
## When using the Sun Studio compiler




# CXXFLAGS = $(ARMA_INCLUDE_FLAG) $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT) 

# all: example1

# example1: example1.cpp
# 	$(CXX) $(CXXFLAGS)  -o $@  $<  $(LIB_FLAGS)


# .PHONY: clean

# clean:
# 	rm -f example1

