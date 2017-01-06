#include "Histogram.h"
#include <stdio.h>
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

Histogram :: Histogram(){
  //Same across all derived processing classes
  //std::vector<arma::mat> data_train; //Check w Jeffrey - or should we just call Mat() constructor here?
  //std::vector<arma::mat> data_test;
  //labels_train = new arma::colvec; //Check w Jeffrey
  //labels_test = new arma::colvec;
  train_has_been_processed = false;
  test_has_been_processed = false;
}

Histogram :: ~Histogram(){
  //Same across all derived processing classes
  //delete data_train;
  //delete data_test;
  //delete labels_train;
  //delete labels_test;
}

//Returns an array sized n X 256 where each row is the pixel counts for one image
int Histogram :: process(){
  std::vector<arma::mat> use_data;
  int flag; //flag for train vs test
  //This is the meat of the processing implementation work
  if(train_has_been_processed == false){
    //Don't let user process already processed data but also no need to throw up lots of errors
    train_has_been_processed=true;
    flag=0;
    use_data=data_train;
    cout<<"Processing training data \n";
  }
  else if(test_has_been_processed == false){
      test_has_been_processed=true;
      flag=1;
      use_data=data_train;
      cout<<"Processing testing data \n";
  }
  //insert algo here
  // Initialize arrays - test and train images have same dimensions
    //# of pixel options won't change
    int pixels=256;
    //Matrix where each row is a histogram for one image - initialize w all zeros
    arma::mat hists(use_data.size(),pixels,fill::zeros);
    //arma::mat currImg();
     //When we have k images we will add another outer loop here to iterate through those
    for(int k=0;k<use_data.size();k++){
	arma::mat currImg=use_data[k];
	int r=currImg.n_rows;
	int c=currImg.n_cols;
     for(int i=0;i<r;i++){
         for(int j=0;j<c;j++){
  	   int temp_pixel=currImg(i,j);
  	   hists(k,temp_pixel)++;
         }
     }
  }
    //Figure out how to return histograms
    if(flag==0){
	data_train=use_data; // They're not the same dimensions is this kosher?
    }
    else if(flag==1){
	data_test=use_data;// They're not the same dimensions is this kosher?
    }
  cout<<"Processing done \n";
  return 0;
}

// Pass the data and the labels
void Histogram :: set_data_train(std::vector<arma::mat > &d_train){
  //Same across all derived processing classes
   data_train = d_train;
}

void Histogram :: set_data_test(std::vector<arma::mat > &d_test){
  //Same across all derived processing classes
  data_test=d_test;
}

void Histogram :: set_labels_train(arma::colvec &l_train){
  //Same across all derived processing classes
  labels_train=l_train;
}

void Histogram :: set_labels_test(arma::colvec &l_test){
  //Same across all derived processing classes
  labels_test=l_test;
}


// Get data and labels
arma::colvec Histogram :: get_labels_train(){
  return(labels_train);
}

arma::colvec Histogram :: get_labels_test(){
  return(labels_test);
}

vector<arma::mat > Histogram :: get_data_train(){
  return(data_train);
}

vector<arma::mat > Histogram ::get_data_test(){
  return(data_test);
}
