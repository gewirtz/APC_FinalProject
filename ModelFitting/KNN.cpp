#include "KNN.h"
#include "Optimizer.h"
#include <cfloat>
#include <cmath> 


using namespace std;
using namespace arma;

KNN::KNN(vector<arma::mat> train, arma::colvec labels, Optimizer *optim){
  this->num_examples = train.size();
  if(num_examples <= 0){
    cerr << "Need an input\n" << endl;
    exit(-1);
  }

  srand(1); //shuffle the elements
  this->x = shuffle(concatenate(train));  //rows contain the ith example, columns contain all instances of a feature
  srand(1); //preserve same shuffling
  this->y = shuffle(labels); //y_i = label of ith training example
  this->optim = optim;

  vector<vec> temp; 
  vec v;
  temp.push_back(v.zeros(x.n_cols)); //plus one because the first item is k
  this->params = temp; // sets all the parameters to 0

  fit();  //fit parameters (average value of each feature for each label)  

  for(int i = 0; i < y.size(); i++){
    this->label_set.insert(y(i));
  }
} 


KNN::~KNN() { 
  //delete [] params; 
}


arma::vec KNN::predict(vector<arma::mat> input){
  mat testset = concatenate(input);
  int k_to_use = k;
  return(internal_predict(testset,x,k_to_use,y));
}

arma::vec KNN::predict_on_subset(arma::mat test, arma::mat train, int k_to_use, arma::vec train_labels){
  return(internal_predict(test,train,k_to_use, train_labels));
}

arma::vec KNN::internal_predict(arma::mat input, arma::mat train, int k_to_use, arma::vec train_labels){
  if(k_to_use<1){
    cerr << "k must be larger than 0" << endl;
    exit(-1)
  }
  vec labels(input.size());
  vec dists(input.size());
  uvec sort_indices;
  vec closest_k_labels(k_to_use);
  ind index_to_use;
  for(int i=0;i<input.size();i++){ // for each item in testset
    for(int j=0;j<train.n_rows;j++){ //get distance to each example in training set
      dists(j) = norm(input.row(i),train.row(j),2);
    }
    sort_indices = sort_index(dists).subvec(0,k_to_use-1); //get the indices of the k closest neighbors
    for(int m=0;m<sort_indices.n_elem;m++){
      index_to_use = sort_indices(m)
      closest_k_labels(m).fill(train_labels(index_to_use))
    }
    //find the mode of closest_k_labels
    int max = 0;
    int mode = -1;
    int cur_label;
    map<int,int> m;
    for (set<int>::iterator n = label_set.begin(); n != label_set.end(); n++) {
      if(cur_index>(train.n_cols)){
        cerr << "Indexing past the assigned length of params in KNN" << endl;
        exit(-1);
      }
      cur_label = *n;
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
    labels(i).fill(mode);
  }
  return(labels);
}

//used for stoch grad descent
vector<vec> KNN::gradient(int k){ 
  if(k < 0 || k >= x.n_rows){
    cerr << "Index " << k << " out of bounds.  Need in range 0, " << x.n_rows << endl;
  }
  vec grad;
  grad = grad.zeros(x.n_cols);
  vec prediction = x.row(k) * params[0]; //Y = X\beta
  double resid = y[k] - prediction[0];  
  
  for(int j = 0; j < x.n_cols;j++){
      grad(j) = resid * x(k,j);
  }  
  vector<vec> v;
  v.push_back(grad);
  return(v);
}

//used for batch gradient descent
vector<vec> KNN::gradient(){
  vec grad;
  grad = grad.zeros(x.n_cols);
  vec predictions = x * params[0]; //Y = X\beta
  vec resid = y - predictions;

  for(int i = 0; i < x.n_rows;  i++){
    for(int j = 0; j < x.n_cols;j++){
      grad(j) += resid(i) * x(i,j);
    }
  }
  vector<vec> v;
  v.push_back(1.0/x.n_rows*grad);
  return(v);
}


void KNN::set_Params(int k, arma::vec p){
  if(k < 0 || k >= params.size()){
    cerr << "Index " << k << " out of bounds.  Need in range 0 " << params.size() << endl;
  }
  params.at(k) = p;
}

vector<vec> KNN::get_Params(){
  return(params);
}

mat KNN::getTrainset(){
  return(x);
}

void KNN::set_k(int k){
  this->k = k;
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
  //for each value in label_set get average value of each feature among test objects
  int cur_index = 0; //start at one because k is at index 0
  for (set<int>::iterator i = label_set.begin(); i != label_set.end(); i++) {
    if(cur_index>(x.n_cols-1)){
      cerr << "Indexing past the assigned length of params in KNN" << endl;
      exit(-1);
    }
    int cur_label = *i;
    uvec indices = find(y==cur_label);
    mat cur_label_samples = x.rows(indices);
    arma::vec mean_vec.zeros(x.n_cols);
    for(int c=0;c<cur_label_samples.n_cols;c++){
      arma::vec cur_feature = x.col(c);
      int avgval = mean(cur_feature);
      mean_vec[c].fill(avgval);
    }
    set_Params(cur_index,mean_vec);
    cur_index++;  
  }
  optim->fitParams(this); //cross-validation for k
  
}
