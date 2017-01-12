#include "KNN.h"
#include "Optimizer.h"
#include <cfloat>
#include <cmath> 
#include <map>


using namespace std;
using namespace arma;

KNN::KNN(vector<arma::mat> train, arma::colvec labels, Optimizer *optim){

  srand(1); //shuffle the elements
  this->x = shuffle(concatenate(train));  //rows contain the ith example, columns contain all instances of a feature
  

  srand(1); //preserve same shuffling
  this->y = shuffle(labels); //y_i = label of ith training example

  mat tempmat;
  tempmat = this->x.rows(0,10000);
  this->x = tempmat;
  vec tempvec;
  tempvec = this->y.subvec(0,10000);
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


arma::vec KNN::predict(vector<arma::mat> input){
  mat testset = concatenate(input);
  if(testset.n_cols != x.n_cols){
    cerr << "Test set must have same number of features as training set" << endl;
    exit(-1);
  }
  mat temp = testset.rows(0,4000);
  testset = temp;
  int k_to_use = params[0](0);
  //create the distance matrix
  mat dists(testset.n_rows,x.n_rows);
  for(int i=0;i<testset.n_rows;i++){
    if(i%1000==0){
      cout << "calculating distance for example " << i << " of " << testset.n_rows -1 << endl;
    }
    for(int j=0;j<x.n_rows;j++){
      dists(i,j) = norm(testset.row(i)-x.row(j),2);
    }
  }
  return(internal_predict(testset,x,k_to_use,y,dists));
}

arma::vec KNN::predict_on_subset(arma::mat test, arma::mat train, int k_to_use, arma::vec train_labels, arma::mat dists){
  return(internal_predict(test,train,k_to_use, train_labels, dists));
}

arma::vec KNN::internal_predict(arma::mat input, arma::mat train, int k_to_use, arma::vec train_labels, const arma::mat dists){
  cout << "internal_predict" << endl;
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
    maxval = dists.row(i).max();
    for(int f=0;f<k_to_use;f++){
      index_to_use = dists_copy.row(i).index_min();
      dists_copy.row(i)(index_to_use) = maxval;
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


mat KNN::concatenate(vector<arma::mat> input){
  int num_rows = input[0].n_rows;
  int num_cols = input[0].n_cols;
  int ex_count = input.size();
  mat data = mat(ex_count,num_rows * num_cols + 1); //includes constant column
  for(int i=0; i<ex_count; i++){
    if(input[i].n_rows!=num_rows || input[i].n_cols!=num_cols ){
      cerr << "Need all input data to have same dimensions\n" << endl;
      exit(-1);
    }

    data(i,0) = 1.0; //regress on constant
  
    for(int j=0;j<num_rows;j++){
      for(int k=0;k<num_cols ; k++){
          data(i,j*num_cols+k+1)=input[i](j,k);
      }
    }
  }
  return(data);
}

void KNN::fit(){

  optim->fitParams(this); //cross-validation for k
  
}

set<int> KNN::getLabelSet(){
  return(label_set);
}
