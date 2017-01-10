#ifndef GRADIENTDESCENT_H_
#define GRADIENTDESCENT_H_

#include "Optimizer.h"

class Model;

class GradientDescent : public Optimizer {
 public:
	// batchSize = 1 is stochastic gradient descent
	//batchSize = 0 is batch gradient descent, alpha is learning rate 
	GradientDescent(int iterations,double alpha, double tol, int batchSize); 
	
	~GradientDescent();
  	void fitParams(Model* m);
 	void setBatchSize(int batchSize);
 	int getBatchSize();
 	std::vector<double> getLastCost();

 private:
 	std::vector<double> cost;  //shows the cost process of the last fit
 	int iterations;
 	double alpha;
 	double tol;
 	int batchSize;
};

#endif  // GRADIENTDESCENT_H_
