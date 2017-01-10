#ifndef CROSSVALIDATION_H_
#define CROSSVALIDATION_H_

#include "Optimizer.h"

class Model;

class CrossValidation : public Optimizer {
 public:
	
	CrossValidation(double range_start, double range_end, double delta=.01); 
	~CrossValidation();
  	void fitParams(Model* m);
  	double param_value;

 private:
	double param_range_start;
	double param_range_end; 
	double delta;
};

#endif  // CROSSVALIDATION_H_
