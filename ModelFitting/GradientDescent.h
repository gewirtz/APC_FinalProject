#ifndef GRADIENTDESCENT_H_
#define GRADIENTDESCENT_H_

#include "Optimizer.h"

class Model;

class GradientDescent : public Optimizer {
 public:
 //is there a better way to pass the gradient 
	GradientDescent(int iterations,double alpha, double tol); 
	~GradientDescent();
  	void fitParams(Model* m, bool fast);
 		
 private:
 	int iterations;
 	double alpha;
 	double tol;
 	double normalizer;
 	void stochasticGradientDescent(Model *m);
 	void batchGradientDescent(Model *m); 
};

#endif  // GRADIENTDESCENT_H_
