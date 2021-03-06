#include "no_processing.h"
#include <stdio.h>
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

//class No_processing : public No_processing{

  //public:

No_processing :: No_processing(){
  //Same across all derived processing classes
  //std::vector<arma::mat> data_train; //Check w Jeffrey - or should we just call Mat() constructor here?
  //std::vector<arma::mat> data_test;
  //labels_train = new arma::colvec; //Check w Jeffrey
  //labels_test = new arma::colvec;
  train_has_been_processed = false;
  test_has_been_processed = false;
}

No_processing :: ~No_processing(){
  //Same across all derived processing classes
  //delete data_train;
  //delete data_test;
  //delete labels_train;
  //delete labels_test;
}


//Process needs to be run twice
//Train will always process first
//This way user cant screw up referencing or feed in random data
int No_processing :: process(){
  //Other versions of this class will have Gaussian stuff implemented here, etc.
  //This is the meat of the processing implementation work
  if(train_has_been_processed == false){
    //Don't let user process already processed data but also no need to throw up lots of errors
    train_has_been_processed=true;
    //cout<<"Processing training data \n";
    //cout<<"Processing done \n";
    return 0;
  }
  if(test_has_been_processed == false){
      test_has_been_processed=true;
      //cout<<"Processing testing data \n";
      //cout<<"Processing done \n";
      return 0;
  }
  return 0;
}

// Pass the data and the labels
void No_processing :: set_data_train(std::vector<arma::mat > &d_train){
  //Same across all derived processing classes
   data_train = d_train;
}

void No_processing :: set_data_test(std::vector<arma::mat > &d_test){
  //Same across all derived processing classes
  data_test=d_test;
}

void No_processing :: set_labels_train(arma::colvec &l_train){
  //Same across all derived processing classes
  labels_train=l_train;
}

void No_processing :: set_labels_test(arma::colvec &l_test){
  //Same across all derived processing classes
  labels_test=l_test;
}


// Get data and labels
arma::colvec No_processing :: get_labels_train(){
  return(labels_train);
}

arma::colvec No_processing :: get_labels_test(){
  return(labels_test);
}

vector<arma::mat > No_processing :: get_data_train(){
  return(data_train);
}

vector<arma::mat > No_processing ::get_data_test(){
  return(data_test);
}
