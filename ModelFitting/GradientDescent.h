#ifndef GRADIENTDESCENT_H_
#define GRADIENTDESCENT_H_

#include "Optimizer.h"

class Model;

class GradientDescent : public Optimizer {
 public:
	// i = 0, stochastic, i = 1, batch, i =2, mixed
	GradientDescent(int iterations,double alpha, double tol, int method, int batchSize); 
	
	~GradientDescent();
  	void fitParams(Model* m);
 	void setMethod(int method);
 	void setBatchSize(int batchSize);
 	int getBatchSize();
 	int getMethod();

 private:
 	int iterations;
 	double alpha;
 	double tol;
 	int batchSize;
 	int method;
 	void stochasticGradientDescent(Model *m);
 	void batchGradientDescent(Model *m); 
 	void mixedGradientDescent(Model*m);  
};

#endif  // GRADIENTDESCENT_H_
