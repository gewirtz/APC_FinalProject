#include "GradientDescent.h"
#include <math.h>
#include "model.h"
#include <armadillo>

using namespace std;
using namespace arma;

GradientDescent::GradientDescent(int iterations,double alpha, double tol, int batchSize){
	this->iterations = iterations; 
	this->alpha = alpha;
	this->tol = tol;

	if(batchSize >= 0){
		this->batchSize = batchSize;
	}
	else{
		cerr << "Need positive number of batches" << endl;
		exit(1);
	}
}

GradientDescent::~GradientDescent(){}

void GradientDescent::fitParams(Model *m){ //fits via mini batch gradient descent
	int length = batchSize;
	if(length == 0){
		length = m->get_num_examples();
	}

	vector<vec> grad;
	vec normalizer;
	int num_params = m->get_Params().size();
	int num_examples = m->get_num_examples();
	
	int pos = 0;
	int upper;
	double update;
	bool finished;
	
	normalizer = normalizer.ones(num_params);

	for(int i = 0; i < iterations; i++){
		pos = 0;

		while(pos < num_examples){
			upper = pos + length;
			if(upper > num_examples){
				upper = num_examples;
			}
			finished = true;
			grad = m->gradient(pos,upper);
			
			for(int j = 0; j < num_params; j++){
				update = norm(grad[j],2);
				cout << "Iteration " << i << " Parameter " << j << " Position " << pos << endl;
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
			pos += length;
		}
		if(finished){
			cout << "Converges on iteration " << i << endl;
			return;
		}
	}
	cerr << "Did not converge in given number of iterations" << endl;
}


void GradientDescent::setBatchSize(int batchSize){
	this->batchSize = batchSize;
}

int GradientDescent::getBatchSize(){
 	return(batchSize);
}
