optimizer = GradientDescent.o
method =   LinearRegression.o 
objects1 =   $(optimizer) $(method) main.o
 
LIB_FLAGS =  -larmadillo
OPT = -O2
CXX = g++
CXXFLAGS = -Wall  $(DEBUG) $(FINAL) $(OPT) 
#-c

all: main

# GradientDescent : $(optimizer)
# 	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB_FLAGS)

# LinearRegression : $(method)
# 	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB_FLAGS)

main : $(objects1)
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

