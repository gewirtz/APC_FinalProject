#include "CrossValidation.h"
#include <math.h>
#include "KNN.h"
#include <armadillo>

using namespace std;
using namespace arma;

//TODO: how to add support when the parameter is not an int
CrossValidation::CrossValidation(double range_start, double range_end, int delta, int nfolds){
	this->param_range_start = range_start;
	this->param_range_end = range_end;
	this->delta = delta;
	this->nfolds = nfolds;
	//to add : some sort of error check making sure the number of params to test isn't too big
}

CrossValidation::~CrossValidation(){}
void CrossValidation::fitParams(Model *m){}

//runs KNN cross validation where k=nfolds to find best parameter in model for the range of possible param vals
void CrossValidation::fitParams(KNN *m){
	//repeat using each division as test set
	mat train = m->getTrainset();
	vec labels = m->getLabels();
	int num_train = train.n_rows;
	int num_steps = (param_range_end-param_range_start)/delta;
	int items_per_fold = num_train/nfolds;
	mat train_set, test_set, train_p1, train_p2;
	vec test_label_set, label_results, train_label_p1, train_label_p2, train_label_set;
	int cur_param, test_start_row, test_end_row, train1_start, train1_end, train2_start, train2_end, need_to_join;
	double param_error,val;
	vec overall_errors(num_steps);
	uvec label_comparison;
	//calculate distance matrix
	mat dists(train.n_rows,train.n_rows);
  	for(int i=0;i<train.n_rows;i++){
  		cout << "calculating distance for example " << i << " of " << train.n_rows -1 << endl;
    	for(int j=0;j<i;j++){
    		val = norm(train.row(i)-train.row(j),2);
      		dists(i,j) = val;
      		dists(j,i) = val;
    	}
    	dists(i,i) = 0.0;
  	}
  	mat dists_to_pass, dists_to_pass_p1, dists_to_pass_p2;
  	//loop through possible values of parameters
	for(int i=0; i<num_steps; i++){ 
		cout <<"step " << i << " of " << num_steps << endl;
		cur_param=param_range_start+(i*delta);
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
			dists_to_pass = dists.rows(test_start_row,test_end_row);
			//now potentially join two submatrices to get the train_set
			if(need_to_join==1){
				cout <<"about to join things" << endl;
				train1_start = (j-1)*items_per_fold;
				train1_end = test_start_row -1;
				train2_start = test_end_row+1;
				train2_end = (j+1)*items_per_fold;
				//cout << "number of rows in train: " << train.n_rows << endl;
				//cout << "extremes trying to access: " << train1_start << " " << train2_end << endl;
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
				dists_to_pass_p1 = dists.cols(train1_start,train1_end);
				dists_to_pass_p2 = dists.cols(train2_start,train2_end);
				dists_to_pass = join_rows(dists_to_pass_p1,dists_to_pass_p2);
			}
			else{
				if(j==0){
					train_set = train.rows(test_end_row+1,num_train-1);
					train_label_set = labels.subvec(test_end_row+1,num_train-1);
					dists_to_pass_p1 = dists_to_pass.cols(test_end_row+1,num_train-1);
					dists_to_pass = dists_to_pass_p1;
				}
				else{
					train_set = train.rows(0,test_start_row-1);
					train_label_set = labels.subvec(0,test_start_row-1);
					dists_to_pass_p1 = dists_to_pass.cols(0,test_start_row-1);
					dists_to_pass = dists_to_pass_p1;
				}
			}
			label_results = m->predict_on_subset(test_set,train_set,cur_param,train_label_set,dists_to_pass);
			//now find how many results were wrong (label_comparison element==0) and add to param_error
			label_comparison = test_label_set==label_results;
			param_error += accu(label_comparison);

		}
		overall_errors(i) = param_error;
	}
	//find which parameter had the smallest error (number of wrong results)
	int best_index = overall_errors.index_min();
	vec param(1);
	param(0) = param_range_start+(best_index*delta);
	m->set_Params(0,param);
	cout << "The best k for KNN was: " << param << endl;
}
