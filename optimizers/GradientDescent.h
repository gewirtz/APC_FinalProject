#ifndef GRADIENTDESCENT_H_
#define GRADIENTDESCENT_H_

#include <Optimizer.h>

class GradientDescent : public Optimizer {
 public:
 //is there a better way to pass the gradient 
	GradientDescent(int iterations, double tol); 
  	void fitParams(Model* m);
 
 private:
 	int iterations;
 	double alpha;
 	double tol;
};

#endif  // OPTIMIZER_H_