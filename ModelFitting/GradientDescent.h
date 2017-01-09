#ifndef GRADIENTDESCENT_H_
#define GRADIENTDESCENT_H_

#include "Optimizer.h"

class Model;

class GradientDescent : public Optimizer {
 public:
 //is there a better way to pass the gradient 
	GradientDescent(int iterations,double alpha, double tol, bool stochastic); 
	~GradientDescent();
  	void fitParams(Model* m);
 	void setType(bool stochastic);
 	bool isStochastic();

 private:
 	int iterations;
 	double alpha;
 	double tol;
 	bool stochastic;
 	void stochasticGradientDescent(Model *m);
 	void batchGradientDescent(Model *m); 
};

#endif  // GRADIENTDESCENT_H_
