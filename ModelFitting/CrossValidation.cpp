#include "CrossValidation.h"
#include <math.h>
#include "KNN.h"
#include "GradientModel.h"


using namespace std;
using namespace arma;

CrossValidation::CrossValidation(double range_start, double range_end, double delta, int nfolds){
	this->param_range_start = range_start;
	this->param_range_end = range_end;
	this->delta = delta;
	if(delta>(range_end-range_start)){
		cerr << "Provided delta is larger than the range for the hyperparam" << endl;
		exit(-1);
	}
	this->nfolds = nfolds;
}

CrossValidation::~CrossValidation(){}

void CrossValidation::fitParams(Model *m){}
void CrossValidation::fitParams(GradientModel *m){}

mat CrossValidation::calculate_dists(mat train){
	mat dists(train.n_rows,train.n_rows);
	double val;
	rowvec rw, rw2;
  	for(int i=0;i<train.n_rows;i++){
  		if(i%1000 == 0){
  			cout << "calculating distance for example " << i << " of " << train.n_rows -1 << endl;
  		}
  		rw = train.row(i);
    	for(int j=0;j<i;j++){
    		rw2 = train.row(j);
    		val = norm(rw-rw2,2);
      		dists(i,j) = val;
      		dists(j,i) = val;
    	}
    	dists(i,i) = 0.0;
  	}
  	return(dists);
}

//runs KNN cross validation where k=nfolds to find best parameter in model for the range of possible param vals
void CrossValidation::fitParams(KNN *m){
	//repeat using each division as test set
	mat train = m->getTrainset();
	if(nfolds>train.n_rows){
		cout << "Number of folds is larger than the number of training examples" << endl;
		exit(-1);
	}
	vec labels = m->getLabels();
	int num_train = train.n_rows;
	int num_steps = (param_range_end-param_range_start)/(int)delta;
	int items_per_fold = num_train/nfolds;
	mat train_set, test_set, train_p1, train_p2;
	vec test_label_set, label_results, train_label_p1, train_label_p2, train_label_set;
	int cur_param, test_start_row, test_end_row, train1_start, train1_end, train2_start, train2_end, need_to_join;
	double param_error;
	vec overall_errors(num_steps);
	uvec label_comparison;
	//calculate distance matrix
	mat dists = calculate_dists(train);
  	mat dists_to_pass, dists_to_pass_p1, dists_to_pass_p2;
  	//loop through possible values of parameters
	for(int i=0; i<num_steps; i++){ 
		cout <<"step " << i+1 << " of " << num_steps << endl;
		cur_param=param_range_start+(i*(int)delta);
		if(cur_param>param_range_end || cur_param<param_range_start){
			cerr << "Current parameter is outside the specified range" << endl;
			exit(-1);
		}
		param_error=0.0;
		//loop through using each fold as test set
		for(int j=0; j<nfolds; j++){
			cout << "fold " << j << endl;
			need_to_join = 1;
			//if the test set is the first fold or last fold, don't need to concatenate mats
			if(j==0 || j==nfolds-1){
				need_to_join = 0;
			}
			if(j==nfolds-1){
				test_end_row = num_train-1;
			}
			//because integer division won't necessarily hit all the training examples
			else{
				test_end_row = ((j+1)*items_per_fold)-1;
			}
			test_start_row = j*items_per_fold;
			test_set = train.rows(test_start_row,test_end_row);
			test_label_set = labels.subvec(test_start_row,test_end_row);
			dists_to_pass = dists.cols(test_start_row,test_end_row);
			
			//now potentially join two submatrices to get the train_set
			if(need_to_join==1){
				train1_start = 0;
				train1_end = test_start_row -1;
				train2_start = test_end_row+1;
				train2_end = num_train-1;
				train_p1 = train.rows(train1_start,train1_end);
				train_p2 = train.rows(train2_start, train2_end);
				train_label_p1 = labels.subvec(train1_start,train1_end);
				train_label_p2 = labels.subvec(train2_start,train2_end);
				train_set = join_cols(train_p1,train_p2);
				train_label_set = vec(train_label_p1.n_elem+train_label_p2.n_elem);
				for(int p = 0; p< train_label_set.n_elem; p++){
					if(p<train_label_p1.n_elem){
						train_label_set(p) = train_label_p1(p);
					}
					else{
						train_label_set(p) = train_label_p2(p-train_label_p1.n_elem);
					}
				}

				dists_to_pass_p1 = dists_to_pass.rows(train1_start,train1_end);
				dists_to_pass_p2 = dists_to_pass.rows(train2_start,train2_end);
				dists_to_pass = join_cols(dists_to_pass_p1,dists_to_pass_p2);
			}
			else{
				if(j==0){
					train_set = train.rows(test_end_row+1,num_train-1);
					train_label_set = labels.subvec(test_end_row+1,num_train-1);
					dists_to_pass_p1 = dists_to_pass.rows(test_end_row+1,num_train-1);
					dists_to_pass = dists_to_pass_p1;
				}
				else{
					train_set = train.rows(0,test_start_row-1);
					train_label_set = labels.subvec(0,test_start_row-1);
					dists_to_pass_p1 = dists_to_pass.rows(0,test_start_row-1);
					dists_to_pass = dists_to_pass_p1;
				}
			}
			label_results = m->predict_on_subset(test_set,train_set,cur_param,train_label_set,dists_to_pass);
			//now find how many results were wrong (label_comparison element==0) and add to param_error
			label_comparison = test_label_set==label_results;
			param_error += test_label_set.n_elem - accu(label_comparison);

		}
		overall_errors(i) = param_error;
	}
	//find which parameter had the smallest error (number of wrong results)
	int best_index = overall_errors.index_min();
	for(int e=0;e<overall_errors.n_elem;e++){
		cout << "The overall error for k=" << param_range_start+(e*(int)delta) << " is " << overall_errors(e) << endl;
	}
	vec param(1);
	param(0) = param_range_start+(best_index*(int)delta);
	m->set_Params(0,param);
	cout << "The best k for KNN was: " << param << endl;
}
