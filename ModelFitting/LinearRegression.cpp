#include "LinearRegression.h"
#include "Optimizer.h"
#include <cfloat>
#include <cmath> 


// initializes a model to choose \beta so as to fit Y = X\beta + \epsilon so as to minimize
// ||Y - X\beta ||_2^2
//TO DO, SYNCH WITH ARI MAP CODE, ADD REGULARIZATION

using namespace std;
using namespace arma;

LinearRegression::LinearRegression(arma::mat train, arma::colvec labels, Optimizer *optim, bool normalize){
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

  vector<vec> temp; 
  vec v;
  temp.push_back(v.zeros(x.n_cols));
  this->params = temp;
  cout << "fitting params " << endl;
  fit();  //fit beta  
} 


LinearRegression::~LinearRegression() {}


vec LinearRegression::predict(arma::mat test){
  if(test.n_cols != this->initial_regressors){
    cout << "Error: train and test must have same number of regressors" << endl;;
    exit(1);
  }
  if(normalize){ //set cols to mean 0 stdev 1
    test = standardize(test);
  }
  vec u;
  test = join_rows(u.ones(test.n_rows),test);

  vec labels = test * params[0]; //non_integer fit
  int closest = 0;
  double distance;
  double temp;

  for(int i=0; i<test.n_rows; i++){ //round it
    distance = DBL_MAX;
    for(int lab : label_set){
      temp = std::abs(labels[i] - lab); //find closest label
      if(temp <= distance){
        distance = temp;
        closest = lab;
      }
    }
    labels[i] = closest;
  }
  return(labels);
}

vec LinearRegression::get_exactParams(){ 
  return(pinv(x.t() * x) * x.t() * y);
}


//gradient between lower and upper
vector<vec> LinearRegression::gradient(int lower, int upper){
  if(lower < 0 || lower >= upper || upper > num_examples){
    cerr << "Lower and upper limits " << lower << " and " << upper << " invalid" << endl;
    cerr << "Need val between 0 and " << num_examples << endl;
    exit(1);
  }
  vector<vec> v;
  vec grad;
  vec predictions;
  vec resid;
  for(int k = 0; k < params.size(); k++){
    grad = grad.zeros(x.n_cols);
    predictions = x * params[k]; //Y = X\beta
    resid = predictions - y;
    for(int i = lower; i < upper;i++){
      for(int j = 0; j < x.n_cols;j++){
        grad(j) += resid(i) * x(i,j);
      }
    }
    v.push_back(1.0/(upper - lower)*grad);
  }
  return(v);
}


void LinearRegression::set_Params(int k, arma::vec p){
  if(k < 0 || k >= params.size()){
    cerr << "Index " << k << " out of bounds.  Need in range 0 " << params.size() << endl;
  }
  params.at(k) = p;
}

vector<vec> LinearRegression::get_Params(){
  return(params);
}

mat LinearRegression::getTrainset(){
  return(x);
}

vec LinearRegression::getLabels(){
  return(y);
}

int LinearRegression::get_num_examples(){
  return(num_examples);
}

double LinearRegression::cost(int lower, int upper, int k){
  vec fits = this->y - x*(params[0]);
  double cost = 0.0;
  for(int i = 0; i < x.n_rows; i++){
    cost += pow(fits[i],2);
  }
  return(1.0/(2*(upper - lower)) * cost);
}

mat LinearRegression::standardize(mat data){
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

void LinearRegression::fit(){
  optim->fitParams(this); //gradient descent
}

set<int> LinearRegression::getLabelSet(){
  return(label_set);
}


