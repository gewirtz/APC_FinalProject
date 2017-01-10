#ifndef CROSSVALIDATION_H_
#define CROSSVALIDATION_H_

#include "Optimizer.h"

class Model;

class CrossValidation : public Optimizer {
 public:
	
	CrossValidation(double range_start, double range_end, double delta=.01, double nfolds=10); 
	~CrossValidation();
  	void fitParams(Model* m);

 private:
	double param_range_start;
	double param_range_end; 
	double delta;
	double nfolds;
};

#endif  // CROSSVALIDATION_H_
