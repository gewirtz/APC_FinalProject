#include "CrossValidation.h"
#include <math.h>
#include "model.h"
#include <armadillo>

using namespace std;
using namespace arma;

//TODO: how to add support when the parameter is not an int
CrossValidation::CrossValidation(double range_start, double range_end, int delta=1, double nfolds=10){
	this->param_range_start = range_start;
	this->param_range_end = range_end;
	this->delta = delta;
	this->nfolds = nfolds;
	//to add : some sort of error check making sure the number of params to test isn't too big
}

CrossValidation::~CrossValidation(){}

//runs KNN cross validation where k=nfolds to find best parameter in model for the range of possible param vals
void CrossValidation::fitParams(Model *m){
	//repeat using each division as test set
	mat train = m->getTrainset();
	vec labels = m->getLabels();
	int num_train = train.n_rows;
	int num_steps = (range_end-range_start)/delta;
	int items_per_fold = num_train/nfolds;
	mat train_set, test_set, train_p1, train_p2;
	vec test_label_set, label_results, train_label_p1, train_label_p2, train_label_set;
	int cur_param, test_start_row, test_end_row, train1_start, train1_end, train2_start, train2_end, need_to_join;
	double param_error;
	vec overall_errors;
	uvec label_comparison;
	//loop through possible values of parameters
	for(int i=0; i<=num_steps; i++){ 
		cur_param=range_start+(i*delta);
		if(cur_param>range_end || cur_param<range_start){
			cerr << "Current parameter is outside the specified range" << endl;
			exit(-1);
		}
		param_error=0.0;
		//loop through using each fold as test set
		for(int j=0; j<nfolds; j++){
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
				test_end_row = ((i+1)*items_per_fold)-1
			}
			test_start_row = i*items_per_fold;
			test_set = train.rows(test_start_row,test_end_row);
			test_label_set = labels.subvec(start_row,end_row);
			//now potentially join two submatrices to get the train_set
			if(need_to_join==1){
				train1_start = (i-1)*items_per_fold;
				train1_end = test_start_row -1;
				train2_start = test_end_row+1;
				train2_end = (i+1)*items_per_fold;
				train_p1 = train.rows(train1_start,train1_end);
				train_p2 = train.rows(train2_start, train2_end);
				train_label_p1 = labels.subvec(train1_start,train1_end);
				train_label_p2 = labels.subvec(train2_start,train2_end);
				train_set = join_cols(train_p1,train_p2);
				train_label_set = vec(train_label_p1.n_elem+train_label_p2.n_elem);
				train_label_set.subvec(0,train_label_p1.n_elem-1).fill(train_label_p1);
				train_label_set.subvec(train_label_p1.n_elem,train_label_set.n_elem).fill(train_label_p2);
			}
			else{
				if(j==0){
					train_set = train.rows(test_end_row+1,num_train-1);
					train_label_set = labels.subvec(test_end_row+1,num_train-1);
				}
				else{
					train_set = train.rows(0,test_start_row-1);
					train_label_set = labels.subvec(0,test_start_row-1);
				}
			}
			label_results = m->predict_on_subset(test_set,train_set,cur_param,train_label_set);
			//now find how many results were wrong (label_comparison element==0) and add to param_error
			label_comparison = test_label_set==label_results;
			param_error += accu(label_comparison);

		}
		overall_errors(i).fill(param_error);
	}
	//find which parameter had the smallest error (number of wrong results)
	int best_index = overall_errors.index_min;
	m->set_k(range_start+(best_index*delta));
}
