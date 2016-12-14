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
	for(int i =0; i < iterations; i++){
		cout << i << endl;
		arma::vec grad = m->gradient();
		if(abs(grad.max()) < tol && abs(grad.min()) < tol ){
			return;
		}
		m->params -= alpha*grad;
	}
	cerr << "Did not converge in given number of iterations" << endl;
}

