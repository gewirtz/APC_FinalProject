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
	bool finished;
	//cout << m->params.size() << endl;
	//cout << m->params.at(0) << endl;
	for(int i =0; i < m->get_params_size(); i++){
		cout << i << endl;
		finished = true;
		for(int j = 0; j < m->params[i].n_cols; j++){
			grad = m->gradient(i);
			update = norm(grad,2);
			if(update > normalizer){
				normalizer = update;
			}
			m->params[i] -= alpha*grad/normalizer;
			//cout << "Iteration " << i << endl;
			//cout << "The update norm is " << update  << endl; 
			//cout << "The maximum gradient element is " << grad.max() << endl;
			//cout << "The minimum gradient element is " << grad.min() << endl;
			if(update  > tol ){
				finished = false;
			}
		}
		if(finished){
			return;
		}
	}
	cerr << "Did not converge in given number of iterations" << endl;
}

