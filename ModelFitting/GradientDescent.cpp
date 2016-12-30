#include "GradientDescent.h"
#include <math.h>
#include "model.h"
#include <armadillo>

using namespace std;
using namespace arma;

GradientDescent::GradientDescent(int iterations, double alpha, double tol){
	this->iterations = iterations;
	this->alpha = alpha;
	this->tol = tol;
	this->normalizer = 1.0;
}

void GradientDescent::fitParams(Model *m){
	arma::vec grad;
	double update;
	for(int i =0; i < iterations; i++){
		grad = m->gradient();
		update = norm(grad,2);
		if(update > normalizer){
			normalizer = update;
		}
		m->params -= alpha*grad/normalizer;
		//cout << "Iteration " << i << endl;
		//cout << "The update norm is " << update  << endl; 
		//cout << "The maximum gradient element is " << grad.max() << endl;
		//cout << "The minimum gradient element is " << grad.min() << endl;
		if(update  < tol ){
			return;
		}
	}
	cerr << "Did not converge in given number of iterations" << endl;
}

