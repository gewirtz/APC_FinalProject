#include "GradientDescent.h"
#include <math.h>
#include "model.h"
#include <armadillo>

using namespace std;
using namespace arma;

GradientDescent::GradientDescent(int iterations, double alpha, double tol, bool stochastic){
	this->iterations = iterations;
	this->alpha = alpha;
	this->tol = tol;
	this->stochastic = stochastic;
}

GradientDescent::~GradientDescent(){}

void GradientDescent::fitParams(Model *m){
	if(stochastic){
		stochasticGradientDescent(m);
	}
	else{
		batchGradientDescent(m);
	}
}

void GradientDescent::setType(bool stochastic){
	this->stochastic = stochastic;
}

bool GradientDescent::isStochastic(){
	return(stochastic);
}

void GradientDescent::stochasticGradientDescent(Model *m){
	int num_examples = m->get_num_examples();
	vector<vec> grad;
	vec normalizer;
	int num_params = m->get_Params().size();

	double update;
	bool finished;
	
	normalizer = normalizer.ones(num_params);
	for(int i = 0; i < iterations; i++){
		finished = true;
		for(int j = 0; j < num_examples; j++){
			grad = m->gradient(j);
			for(int k = 0; k < num_params; k++){
				cout << "iteration "<< i << " example " << j << " parameter " << k << endl;
				update = norm(grad[k],2);
				cout << "The update norm is " << update  << endl; 
				cout << "The maximum gradient element is " << grad[k].max() << endl;
				cout << "The minimum gradient element is " << grad[k].min() << endl;
				if(update > normalizer[k]){
					normalizer[k] = update;
				}
				m->set_Params(k, m->get_Params()[k] - alpha*grad[k]/normalizer[k]);
				if(update > tol){
					finished = false;
				}
			}
		}
		if(finished){
			cout << "Converges on iteration " << i << endl;
			return;
		}
	}
	cerr << "Did not converge in given number of iterations" << endl;
}


void GradientDescent::batchGradientDescent(Model *m){
	vector<vec> grad;
	vec normalizer;
	int num_params = m->get_Params().size();

	double update;
	bool finished;
	//int reset = min(500,iterations/10);
	normalizer = 1.0 * normalizer.ones(num_params);

	for(int i = 0; i < iterations; i++){
		grad = m->gradient();
		finished = true;
		for(int j = 0; j < num_params; j++){
			//cout << "iteration "<< i << " parameter " << j << endl;

			update = norm(grad[j],2);
			cout << "Gradient length " << grad[j].size() << endl;
			cout << "The update norm is " << update  << endl; 
			cout << "The maximum gradient element is " << grad[j].max() << endl;
			cout << "The minimum gradient element is " << grad[j].min() << endl;

			if(update > normalizer[j]){
				normalizer[j] = update;
			}

			m->set_Params(j, m->get_Params()[j] - alpha*grad[j]/normalizer[j]);
			if(update > tol){
				finished = false;
			}
		}
		if(finished){
			cout << "Converges on iteration " << i << endl;
			return;
		}
	}
	cerr << "Did not converge in given number of iterations" << endl;
}

