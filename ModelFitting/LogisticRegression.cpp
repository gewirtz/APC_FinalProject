#include "LogisticRegression.h"
#include "Optimizer.h"
#include <cfloat>
#include <cmath> 


// initializes a model to classify via the probability model 
// P(Y=j|X) = \frac{\exp(\theta_j*X_i)}/{1+\sum_k\exp(\theta_k*X_i)} 
//TO DO, SYNCH WITH ARI MAP CODE, ADD REGULARIZATION

using namespace std;
using namespace arma;


LogisticRegression::LogisticRegression(arma::mat train, arma::colvec labels, Optimizer *optim, bool normalize){
  this->initial_regressors = train.n_cols;
  this->normalize = normalize;
  this->trained = false;

  //mat tempmat;
  //tempmat = this->x.rows(0,10000);
  //this->x = tempmat;
  //vec tempvec;
  //tempvec = this->y.subvec(0,10000);
  //this->y = tempvec; 
  if(normalize){ //set cols to mean 0 stdev 1
    train = standardize(train);
    if(train.n_cols == 0){
      cerr << "Cannot have all constant regressors" << endl;
      exit(1); 
    }
  }

  vec u;
  u = u.ones(train.n_rows);
  train = join_rows(u,train);
  srand(524); //shuffle the elements
  this->x = shuffle(train);  //rows contain the ith example, columns contain all instances of a feature
  srand(524); //preserve same shuffling
  this->y = shuffle(labels); //y_i = label of ith training example
  this->optim = optim;

  this->num_examples = x.n_rows;
  if(num_examples <= 0){
    cerr << "Need an input\n" << endl;
    exit(-1);
  }


  for(int i = 0; i < y.size(); i++){
    this->label_set.insert(y(i));
  }

  set_ovr_labels();
  vector<vec> temp; 
  vec v;
  for(int i = 0; i < label_set.size();i++){
    temp.push_back(v.zeros(x.n_cols));
  }
  this->params = temp;


  cout << "fitting params " << endl;
  fit();  //fit beta  
} 

LogisticRegression::~LogisticRegression() {}


void LogisticRegression::set_ovr_labels(){ //memoization
  vector<int> temp = vector<int>(num_examples);
  for(int k = 0; k < label_set.size(); k++){
    //create one v rest label set for label k
    for(int i = 0; i < num_examples;  i++){
      if(y[i] == k){
        temp[i] = 1;
      }
      else{
        temp[i] = 0;
      }
    }
    ovr_labels.push_back(temp);
  }
}

vec LogisticRegression::predict(arma::mat test){
  if(test.n_cols != this->initial_regressors){
    cout << "Error: train and test must have same number of regressors" << endl;;
    exit(1);
  }
  if(normalize){ //set cols to mean 0 stdev 1
    test = standardize(test);
  }
  vec u;
  test = join_rows(u.ones(test.n_rows),test);

  vec labels(test.n_rows);
  int fitted_val = -1;
  vector<double> label_likelihood(label_set.size());
  vec temp;
  double max_prob;
  for(int i = 0; i < test.n_rows; i++){
    max_prob = 0.0;

    //for(int k = 0; k < label_set.size();k++){
    for(int k : label_set){
      temp = test.row(i) * params[k];
      label_likelihood[k] = 1.0/(1.0+exp(-temp[0]));
      if(label_likelihood[k] > max_prob){
        max_prob = label_likelihood[k];
        fitted_val = k;
    //    fitted_val = label_set[k]; //get this to synch with ari's code
      } 
    }
    labels[i] = fitted_val;
  }
  return(labels);
}

vector<vec> LogisticRegression::gradient(int lower, int upper){ //one v rest fit
   //cout << "Gradient" << endl;
  if(lower < 0 || lower >= upper || upper > num_examples){
    cerr << "Lower and upper limits " << lower << " and " << upper << " invalid" << endl;
    cerr << "Need val between 0 and " << num_examples << endl;
    exit(1);
  }
  vector<vec> v;
  vec grad; 
  vec probs;
  for(int k = 0; k < label_set.size(); k++){

    grad = grad.zeros(x.n_cols); //reset grad
    probs = 1.0/(1.0+exp(-x*params[k])); 
    
    for(int i = 0; i < x.n_cols; i++){ //i=1...nparams
      //cout << i << endl;
      for(int j = lower; j < upper; j++){ // j=1...examples
        /*cout << j << endl;
        cout << "ovr" << endl;
        cout << ovr_labels[k][j] << endl;
        cout << "probs" << endl;
        cout << probs(j) << endl; 
        cout<<"x" << endl;
        cout << x(j,i) << endl;*/
        grad(i) += (ovr_labels[k][j] - probs(j)) * x(j,i);
      }
    }
    v.push_back(-1.0/(upper - lower)*grad);
  }
  return(v); //maximizing likelihood is equivalent to minimizing negative likelihood
}


void LogisticRegression::set_Params(int k, arma::vec p){
  if(k < 0 || k >= params.size()){
    cerr << "Index " << k << " out of bounds.  Need in range 0 " << params.size() << endl;
  }
  params.at(k) = p;
}


vector<vec> LogisticRegression::get_Params(){
  return(params);
}

int LogisticRegression::get_num_examples(){
  return(num_examples);
}

double LogisticRegression::cost(int lower, int upper, int k){
  vec hyp = 1.0/(1.0+exp(-x*params[k])); //logistic hypothesis function
  double cost = 0.0;
  for(int i = lower; i < upper; i++){
    cost += -y[i] *hyp[i] - (1-y[i])*log(1-hyp[i]);
  }
  return(1.0/((upper - lower)) * cost);
}

mat LogisticRegression::getTrainset(){
  return(x);
}


mat LogisticRegression::standardize(mat data){
  if(!this->trained){
    cout << "Training standardization " << endl;
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


vec LogisticRegression::getLabels(){
  return(y);
}

void LogisticRegression::fit(){
  optim->fitParams(this);
}

set<int> LogisticRegression::getLabelSet(){
  return(label_set);
}



