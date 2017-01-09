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

void GradientDescent::fitParams(Model *m, bool fast){
	if(fast){
		stochasticGradientDescent(m);
	}
	else{
		batchGradientDescent(m);
	}
}

void GradientDescent::stochasticGradientDescent(Model *m){
	int num_examples = m->get_num_examples;
	vector<vec> grad;
	vec normalizer;
	int num_params = m->get_Params.size();

	double update;
	bool finished;
	
	normalizer = normalizer.ones(num_params);
	for(int i = 0; i < iterations; i++){
		finished = true;
		for(int j = 0; j < num_examples; j++){
			grad = m->gradient(j);
			for(int k = 0; k < num_params; k++){
				update = norm(grad[k],2);
				if(update > normalizer[k]){
					normalizer[k] = update;
				}
				m->set_Params(k, m->get_Params()[k] - alpha*grad[k]/normalizer[k]);
			}
			if(update > tol){
				finished = false;
			}
		}
		if(finished){
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
	normalizer = normalizer.ones(num_params);

	for(int i = 0; i < iterations; i++){
		grad = m->gradient();
		finished = true
		for(int j = 0; j < num_params; k++){
			update = norm(grad,2);
			if(update > normalizer[j]){
				normalizer[j] = update;
			}
			m->set_Params(j, m->get_Params()[j] - alpha*grad[j]/normalizer[j]);
			if(update > tol){
				finished = false;
			}
		}
		if(finished){
			return;
		}
	}
	cerr << "Did not converge in given number of iterations" << endl;
}
/*void GradientDescent::batchGradientDescent(Model *m){
	vec grad;
	double update;
	bool finished;
	//int reset = min(500,iterations/10);
	for(int i =0; i < m->get_Params().size(); i++){
		normalizer = 1.0;
		finished = true;
		cout << m->get_Params().size() << endl;
		for(int j = 0; j < iterations; j++){
			*if(iterations % reset == 0){
				normalizer = 1.0;
			}*
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
}*/
