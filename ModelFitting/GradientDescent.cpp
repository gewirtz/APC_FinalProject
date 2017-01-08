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

GradientDescent::~GradientDescent(){}

void GradientDescent::fitParams(Model *m){
	cout << "Descent" << endl;
	vec grad;
	double update;
	bool finished;
	//cout << m->params.size() << endl;
	//cout << m->params.at(0) << endl;
	for(int i =0; i < m->get_Params().size(); i++){
		cout << i << endl;
		finished = true;
		cout << m->get_Params().size() << endl;
		for(int j = 0; j < iterations; j++){
			grad = m->gradient(i);
			update = norm(grad,2);
			if(update > normalizer){
				normalizer = update;
			}
			m->set_Params(i, m->get_Params()[i] - alpha*grad/normalizer);
			cout << "Iteration " << i << "  " << j << endl;
			cout << "The update norm is " << update  << endl; 
			cout << "The maximum gradient element is " << grad.max() << endl;
			cout << "The minimum gradient element is " << grad.min() << endl;
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

