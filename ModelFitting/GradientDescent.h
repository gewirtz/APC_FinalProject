#ifndef GRADIENTDESCENT_H_
#define GRADIENTDESCENT_H_

#include "Optimizer.h"
#include "GradientModel.h"

class GradientDescent : public Optimizer {
 public:
	// batchSize = 1 is stochastic gradient descent
	//batchSize = 0 is batch gradient descent, alpha is learning rate 
	GradientDescent(int iterations,double alpha, double tol, int batchSize); 
	
	~GradientDescent();
  	void fitParams(GradientModel* m);
  	void fitParams(Model *m);
	void fitParams(KNN *m);

 	void setBatchSize(int batchSize);
 	int getBatchSize();
 	std::vector<std::vector<double>> getLastCost();

 private:
 	std::vector<std::vector<double>> cost;  //shows the cost process 
 	std::vector<double> last_cost;
 	int iterations;
 	double alpha;
 	double tol;
 	int batchSize;
};

#endif  // GRADIENTDESCENT_H_
