#ifndef CROSSVALIDATION_H_
#define CROSSVALIDATION_H_

#include "Optimizer.h"

class Model;

class CrossValidation : public Optimizer {
 public:
	
	CrossValidation(double range_start, double range_end, int delta , int nfolds ); 
	~CrossValidation();
  	void fitParams(Model* m);

 private:
	double param_range_start;
	double param_range_end; 
	int delta;
	int nfolds;
};

#endif  // CROSSVALIDATION_H_
