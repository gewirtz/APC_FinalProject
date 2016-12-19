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
}

void GradientDescent::fitParams(Model *m){
	arma::vec grad;
	arma::vec old_params;
	for(int i =0; i < iterations; i++){
		grad = m->gradient();
		old_params = m->params;
		m->params -= alpha*grad;
		cout << "Iteration " << i << endl;
		cout << "The update norm is " << norm(old_params- m->params,2)  << endl; 
		cout << "The maximum gradient element is " << grad.max() << endl;
		cout << "The minimum gradient element is " << grad.min() << endl;
		if(norm(old_params - m->params,2)  < tol ){
			return;
		}
	}
	cerr << "Did not converge in given number of iterations" << endl;
}

