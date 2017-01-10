#include "GradientDescent.h"
#include <math.h>
#include "model.h"
#include <armadillo>

using namespace std;
using namespace arma;

GradientDescent::GradientDescent(int iterations,double alpha, double tol, int method, int batchSize){
	this->iterations = iterations;
	this->alpha = alpha;
	this->tol = tol;
	if(method == 0 || method == 1 || method == 2){
		this->method = method;
	}
	else{
		cerr << "Invalid choice of gradient descent method " << endl;
		cerr << "Please enter 0,1, or 2" << endl;
		exit(1);
	}
	if(batchSize > 0){
		this->batchSize = batchSize;
	}
	else{
		cerr << "Need positive number of batches" << endl;
		exit(1);
	}
}

GradientDescent::~GradientDescent(){}

void GradientDescent::fitParams(Model *m){
	if(method == 0){
		stochasticGradientDescent(m);
	}
	else if(method == 1) {
		batchGradientDescent(m);
	}
	else{
		mixedGradientDescent(m);
	}
}

void GradientDescent::setMethod(int method){
	if(method == 0 || method == 1 || method == 2){
		this->method = method;
	}
	else{
		cerr << "Invalid choice of gradient descent method " << endl;
		cerr << "Please enter 0,1, or 2" << endl;
		exit(1);
	}
}

int GradientDescent::getMethod(){
	return(method);
}

void GradientDescent::setBatchSize(int batchSize){
	this->batchSize = batchSize;
}

int GradientDescent::getBatchSize(){
 	return(batchSize);
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
			grad = m->gradient(j,j+1);
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
	normalizer = normalizer.ones(num_params);

	for(int i = 0; i < iterations; i++){
		grad = m->gradient(0,num_params);
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



void GradientDescent::mixedGradientDescent(Model *m){
	vector<vec> grad;
	vec normalizer;
	int num_params = m->get_Params().size();
	int num_examples = m->get_num_examples();
	//int remainder = num_examples % batchSize;
	//int num_batches = (num_examples - remainder) / batchSize;
	int pos = 0;
	int upper;
	double update;
	bool finished;
	//int reset = min(500,iterations/10);
	normalizer = normalizer.ones(num_params);

	for(int i = 0; i < iterations; i++){
		pos = 0;

		while(pos < num_examples){
			upper = pos + batchSize;
			if(upper > num_examples){
				upper = num_examples;
			}
			finished = true;
			grad = m->gradient(pos,upper);
			
			for(int j = 0; j < num_params; j++){
				update = norm(grad[j],2);
				
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
			pos += batchSize;
		}
		if(finished){
			cout << "Converges on iteration " << i << endl;
			return;
		}
	}
	cerr << "Did not converge in given number of iterations" << endl;
}


