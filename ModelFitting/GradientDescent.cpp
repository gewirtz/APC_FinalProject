#include "GradientDescent.h"
#include <math.h>
#include "GradientModel.h"
#include <armadillo>

using namespace std;
using namespace arma;



GradientDescent::GradientDescent(int iterations,double alpha, double tol, int batchSize){
	this->iterations = iterations; 
	this->alpha = alpha;
	this->tol = tol;
	vector<vector<double>> v;
	this->cost = v; 
	if(batchSize >= 0){
		this->batchSize = batchSize;
	}
	else{
		cerr << "Need positive number of batches" << endl;
		exit(1);
	}
}


GradientDescent::~GradientDescent(){}

void GradientDescent::fitParams(Model *m){
	cerr << "Only call GradientDescent on objects of type GradientModel" << endl;
	exit(1);
}

void GradientDescent::fitParams(KNN *m){
	cerr << "Only call GradientDescent on objects of type GradientModel" << endl;
	exit(1);
}


void GradientDescent::fitParams(GradientModel *m){ //fits via mini batch gradient descent
	int length = batchSize;
	if(length == 0){
		length = m->get_num_examples();
	}

	int num_params = m->get_Params().size();
	int num_examples = m->get_num_examples();
	
	//initialize learning rate so it runs a seperate gradient descent for each param
	vec alphas;
	alphas = alpha * alphas.ones(num_params); 
	
	//reset costs
	vector<vector<double>> v(num_params);
	for(int i = 0; i < num_params; i++){
		v[i].push_back(m->cost(0,num_examples,i)); //initialized to be the cost of the entire fit
	}
	this->cost = v;  
	vector<vec> grad;

	//variables needed for keeping track of costs
	vec last_cost;
	last_cost = last_cost.ones(num_params) * 10e99;
	double temp_cost;

	//iterator varaibles
	int pos = 0;
	int upper;
	double update;
	bool finished;
	double thresh = 1.0;

	for(int i = 0; i < iterations; i++){
		if(i > iterations/2){
			thresh = 10e-1;
		}
		if(i > 3.0 *iterations/4){
			thresh = 10e-5;
		}


		pos = 0;

		while(pos < num_examples){
			upper = pos + length;
			if(upper > num_examples){
				upper = num_examples;
			}
			finished = true;
			grad = m->gradient(pos,upper);
			for(int j = 0; j < num_params; j++){
				//check for convergence
				update = norm(grad[j],2);
				if(update > tol){
					finished = false;
				}

				m->set_Params(j, m->get_Params()[j] - alphas[j]*grad[j]);
				temp_cost = m->cost(pos,upper,j);
				cost[j].push_back(temp_cost);
				
				if(temp_cost - thresh> last_cost[j]){ //bold driver method for updating params
					m->set_Params(j, m->get_Params()[j] + alphas[j]*grad[j]); //undo weight change
					alphas[j] *= .5; //reduce alpha
				}

				else{ 
					alphas[j] *= 1.05;
					last_cost[j] = temp_cost;
				}

				if(j == 0){
					cout << "Iteration " << i << " Parameter " << j << " Position " << pos << endl;
					cout << "The update norm is " << update  << endl; 
					cout << "The maximum gradient element is " << grad[j].max() << endl;
					cout << "The minimum gradient element is " << grad[j].min() << endl;
					cout << "Alpha is " << alphas[j] << endl;
					cout << "The cost is " << temp_cost << endl;
					cout << "The last cost was " << last_cost[j] << endl;
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



/*
void mixedBatchGradientDescent(Model *m, int k){

}
*/
/*
void GradientDescent::fitParams(Model *m){ //fits via mini batch gradient descent
	int length = batchSize;
	if(length == 0){
		length = m->get_num_examples();
	}
	vec normalizer;
	int num_params = m->get_Params().size();
	int num_examples = m->get_num_examples();
	
	//reset costs
	vector<vector<double>> v(num_params);
	this->cost = v;  
	vector<vec> grad;

	int pos = 0;
	int upper;
	double update;
	bool finished;
	double temp_cost;

	for(int i = 0; i < iterations; i++){

		normalizer = normalizer.ones(num_params);
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
				if(j == 0){
					cout << "Iteration " << i << " Parameter " << j << " Position " << pos << endl;
					cout << "The update norm is " << update  << endl; 
					cout << "The maximum gradient element is " << grad[j].max() << endl;
					cout << "The minimum gradient element is " << grad[j].min() << endl;
				}
				if(update > normalizer[j]){
					normalizer[j] = update;
				}
				if(j == 0){
					cout << "The normalizer is " << normalizer[j] << endl;
				}
				m->set_Params(j, m->get_Params()[j] - alpha*grad[j]/normalizer[j]);
				temp_cost = m->cost(pos,upper,j);
				cost[j].push_back(temp_cost);
				if(j == 0){
					cout << "The cost is " << temp_cost << endl;
				}
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
*/

void GradientDescent::setBatchSize(int batchSize){
	this->batchSize = batchSize;
}

int GradientDescent::getBatchSize(){
 	return(batchSize);
}

vector<vector<double>> GradientDescent::getLastCost(){
	return cost;
}
