#include "KNN.h"
#include "Optimizer.h"
#include <cfloat>
#include <cmath> 
#include <map>


using namespace std;
using namespace arma;

KNN::KNN(arma::mat train, arma::colvec labels, Optimizer *optim, bool normalize){
  this->initial_regressors = train.n_cols;
  this->normalize = normalize;
  this->trained = false;
  if(normalize){ //set cols to mean 0 stdev 1
    train = standardize(train);
    if(train.n_cols == 0){
      cerr << "Cannot have all constant regressors" << endl;
      exit(1); 
    }
  }

  srand(1); //shuffle the elements
  this->x = shuffle(train);  //rows contain the ith example, columns contain all instances of a feature
  

  srand(1); //preserve same shuffling
  this->y = shuffle(labels); //y_i = label of ith training example

  mat tempmat;
  tempmat = this->x.rows(0,5000);
  this->x = tempmat;
  vec tempvec;
  tempvec = this->y.subvec(0,5000);
  this->y = tempvec;
  this->num_examples = this->x.n_rows;
  if(num_examples <= 0){
    cerr << "Need an input\n" << endl;
    exit(-1);
  }

  this->optim = optim;
  for(int i = 0; i < y.size(); i++){
    this->label_set.insert(y(i));
  }
  vector<vec> temp; 
  vec v(1);
  temp.push_back(v.zeros());
  this->params = temp; // sets all the parameters to 0
  
  fit();  //fit parameters (average value of each feature for each label)  
} 


KNN::~KNN() { 
  //delete [] params; 
}


arma::vec KNN::predict(mat testset){
  if(testset.n_cols != this->initial_regressors){
    cout << "Error: train and test must have same number of features" << endl;;
    exit(1);
  }
  if(normalize){ //set cols to mean 0 stdev 1
    mat temp = standardize(testset);
    testset = temp;
  }
  mat temp = testset.rows(0,4000);
  testset = temp;
  int k_to_use = params[0](0);
  //create the distance matrix
  mat dists(x.n_rows,testset.n_rows);
  rowvec rw, rw2;
  for(int i=0;i<x.n_rows;i++){
    if(i%1000==0){
      cout << "calculating distance for example " << i << " of " << testset.n_rows -1 << endl;
    }
    rw = x.row(i);
    for(int j=0;j<testset.n_rows;j++){
      rw2 = testset.row(j);
      dists(i,j) = norm(rw-rw2,2);
    }
  }
  return(internal_predict(testset,x,k_to_use,y,dists));
}

arma::vec KNN::predict_on_subset(arma::mat test, arma::mat train, int k_to_use, arma::vec train_labels, arma::mat dists){
  return(internal_predict(test,train,k_to_use, train_labels, dists));
}

arma::vec KNN::internal_predict(arma::mat input, arma::mat train, int k_to_use, arma::vec train_labels, const arma::mat dists){
  arma::mat dists_copy = dists;
  if(k_to_use<1){
    cerr << "k must be larger than 0" << endl;
    exit(-1);
  }
  if(k_to_use>train.n_rows){
    cerr << "k must be no greater than the number of training examples";
    exit(-1);
  }
  vec labels(input.n_rows);
  vec closest_k_labels(k_to_use);
  uword index_to_use;
  double maxval;

  for(int i=0;i<input.n_rows;i++){ // for each item in testset
    maxval = dists.col(i).max();
    for(int f=0;f<k_to_use;f++){
      index_to_use = dists_copy.col(i).index_min();
      dists_copy.col(i)(index_to_use) = maxval;
      closest_k_labels(f) = train_labels(index_to_use);
    }
    
    //find the mode of closest_k_labels
    int max = 0;
    int mode;
    int cur_label;
    map<int,int> m;
    for(int n = 0; n<k_to_use; n++){
      cur_label = closest_k_labels(n);
      if(m.find(cur_label) != m.end()){
        m[cur_label]++;
      }
      else{
        m[cur_label] = 1;
      }
      if(m[cur_label]<1){
        cerr << "The count of neighbors with a label cannot be less than 1" << endl;
        exit(-1);
      }
      if(m[cur_label] > max){
        max = m[cur_label];
        mode = cur_label;
      }
    }
    labels(i) = mode;
  }
  return(labels);
}



void KNN::set_Params(int k, arma::vec p){
  if(k !=0){
    cerr << "Index " << k << " out of bounds while trying to set params" << endl;
  }
  params[k] = p;
}

vector<vec> KNN::get_Params(){
  return(params);
}

mat KNN::getTrainset(){
  return(x);
}

vec KNN::getLabels(){
  return(y);
}

int KNN::get_num_examples(){
  return(num_examples);
}



void KNN::fit(){

  optim->fitParams(this); //cross-validation for k
  
}

mat KNN::standardize(mat data){
  if(!this->trained){
    this->trained = true; //memoize results so as to not run in predict
    this->tr_means = mean(data,0);
    this->tr_stdev = stddev(data,0,0);
    this->remove = vector<bool>(data.n_cols);
    int count = 0;
    for(int j = 0; j < data.n_cols; j++){
      if(tr_stdev[j]> 0.001){ 
          remove[j] = false; //remove constant rows
          count += 1;
        }
        else{ //prevent nans
          remove[j] = true;
        }
    }
    this->num_regressors = count;
  }

  //standardization

  mat new_data(data.n_rows, this->num_regressors);
  int pos;
  int j;
  for(int i = 0; i < new_data.n_rows;i++){
    pos = 0;
    j = 0;
    while(j < new_data.n_cols){
      if(remove[pos]){
        pos += 1;
      }
      else{
        new_data(i,j) = (data(i,pos) - tr_means[pos])/tr_stdev[pos];
        j += 1;
        pos+= 1;
      }
    }
  }
  return(new_data);
}


set<int> KNN::getLabelSet(){
  return(label_set);
}
