#include "gaussian_smoothing.h"
#include <stdio.h>
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

Gaussian_smoothing :: Gaussian_smoothing(){
  //Same across all derived processing classes
  //std::vector<arma::mat> data_train; //Check w Jeffrey - or should we just call Mat() constructor here?
  //std::vector<arma::mat> data_test;
  //labels_train = new arma::colvec; //Check w Jeffrey
  //labels_test = new arma::colvec;
  train_has_been_processed = false;
  test_has_been_processed = false;
}

Gaussian_smoothing :: ~Gaussian_smoothing(){
  //Same across all derived processing classes
  //delete data_train;
  //delete data_test;
  //delete labels_train;
  //delete labels_test;
}

int Gaussian_smoothing :: process(){
  std::vector<arma::mat> use_data;
  int flag; //flag for train vs test data
  //This is the meat of the processing implementation work
  if(train_has_been_processed == false){
      //Don't let user process already processed data but also no need to throw up lots of errors
      train_has_been_processed=true;
      flag=0;
      use_data=data_train;
      //cout<<"Processing training data \n";
  }
  else if(test_has_been_processed == false){
      test_has_been_processed=true;
      flag=1;
      use_data=data_test;
      //cout<<"Processing testing data \n";
  }
  //Using a 5X5 filter implying a border of 2
  int filter_size=5;
  int bord=2;
  //Filter components - using seperability of Gaussian to allow for 1X5 and 5X1 filter
  float a0=0.0625;
  float a1=0.25;
  float a2=0.375;
  float a3=0.25;
  float a4=0.0625;
  //Initialize filter arrays
  //Using sliding window approach to do row and column convulutions simultaneously
  arma::mat currImg=use_data[0];
  int c=currImg.n_cols; //Number of columns of image
  int r=currImg.n_rows; //Number of rows of image
  std::vector<float> conv_h0(c); //Two cells up
  std::vector<float> conv_h1(c); //One cell up
  std::vector<float> conv_h2(c); //Row being processed
  std::vector<float> conv_h3(c); //One cell down
  std::vector<float> conv_h4(c); //Two cells down

  //Initialize larger array for mirror boundary conditions
  arma::mat tempImg((r+2*bord),(c+2*bord));
  for(int k=0;k<use_data.size();k++){
     currImg=use_data[k];
  //Set boundary conditions
  for(int j=bord; j<(c+bord);j++){ //Rows excluding corners
      tempImg(0,j)=currImg(1,j-bord);
      tempImg(1,j)=currImg(0,j-bord);
      tempImg(r+bord+1,j)=currImg(r-2,j-bord);
      tempImg(r+bord,j)=currImg(r-1,j-bord);
  }
  for(int i=bord; i<(r+bord);i++){ //Cols excluding corners
      tempImg(i,0)=currImg(i-bord,1);
      tempImg(i,1)=currImg(i-bord,0);
      tempImg(i,c+bord)=currImg(i-bord,c-1);
      tempImg(i,c+bord+1)=currImg(i-bord,c-2);
  }
  //Set corners - mirroring diagonally
  //Top left corner
  tempImg(0,1)=currImg(0,1);
  tempImg(1,0)=currImg(1,0);
  tempImg(0,0)=currImg(1,1);
  tempImg(1,1)=currImg(0,0);
  //Top right corner
  tempImg(0,c+bord)=currImg(0,c-2);
  tempImg(1,c+bord)=currImg(0,c-1);
  tempImg(0,c+bord+1)=currImg(1,c-2);
  tempImg(1,c+bord+1)=currImg(1,c-1);
  //Bottom left corner
  tempImg(r+bord,0)=currImg(r-2,0);
  tempImg(r+bord,1)=currImg(r-1,0);
  tempImg(r+2*bord-1,0)=currImg(r-2,1);
  tempImg(r+2*bord-1,1)=currImg(r-1,1);
  //Bottom right corner
  tempImg(r+2*bord-1,c+2*bord-1)=currImg(r-2,c-2);
  tempImg(r+bord,c+2*bord-1)=currImg(r-2,c-1);
  tempImg(r+2*bord-1,c+bord)=currImg(r-1,c-2);
  tempImg(r+bord,c+bord)=currImg(r-1,c-1);

  //Copy center of matrix
  for(int i=bord;i<(r+bord);i++){
      for(int j=bord;j<(c+bord);j++){
	  tempImg(i,j)=currImg(i-bord,j-bord);
      }
  }

  //Sliding window - starting in top left corner
  //Horizontal convolution arrays

  for(int j=bord;j<(c+bord);j++){ //Going across columns to get first 5 convolution arrays
      //conv_h0[j-bord]=a0*tempImg(0,j-2)+a1*tempImg(0,j-1)+a2*tempImg(0,j)+a3*tempImg(0,j+1)+a4*tempImg(0,j+2); //Two rows above
      conv_h1[j-bord]=a0*tempImg(0,j-2)+a1*tempImg(0,j-1)+a2*tempImg(0,j)+a3*tempImg(0,j+1)+a4*tempImg(0,j+2); //One row above
      conv_h2[j-bord]=a0*tempImg(1,j-2)+a1*tempImg(1,j-1)+a2*tempImg(1,j)+a3*tempImg(1,j+1)+a4*tempImg(1,j+2); //Current row
      conv_h3[j-bord]=a0*tempImg(2,j-2)+a1*tempImg(2,j-1)+a2*tempImg(2,j)+a3*tempImg(2,j+1)+a4*tempImg(2,j+2); //One row below
      conv_h4[j-bord]=a0*tempImg(3,j-2)+a1*tempImg(3,j-1)+a2*tempImg(3,j)+a3*tempImg(3,j+1)+a4*tempImg(3,j+2); //Two rows below
  }
  for(int i=bord;i<(r+bord);i++){ //Vertical convolutions and moving down rows
      //Reset all horizontal convolutions and calc new one
      conv_h0=conv_h1;
      conv_h1=conv_h2;
      conv_h2=conv_h3;
      conv_h3=conv_h4;
      for(int j=bord;j<(c+bord);j++){ //Going across columns to get first 5 convolution arrays
	  conv_h4[j-2]=a0*tempImg(i+2,j-2)+a1*tempImg(i+2,j-1)+a2*tempImg(i+2,j)+a3*tempImg(i+2,j+1)+a4*tempImg(i+2,j+2); //Two rows below
      }
      for(int j=bord;j<(c+bord);j++){
	  currImg(i-bord,j-bord)=a0*conv_h0[j-bord]+a1*conv_h1[j-bord]+a2*conv_h2[j-bord]+a3*conv_h3[j-bord]+a4*conv_h4[j-bord];
      }
  }
  }
  if(flag==0){
      data_train=use_data;
  }
  else if(flag==1){
      data_test=use_data;
  }

  //cout<<"Processing done \n";
  return 0;
}

// Pass the data and the labels
void Gaussian_smoothing :: set_data_train(std::vector<arma::mat > &d_train){
  //Same across all derived processing classes
  data_train = d_train;
}

void Gaussian_smoothing :: set_data_test(std::vector<arma::mat > &d_test){
  //Same across all derived processing classes
  data_test=d_test;
}

void Gaussian_smoothing :: set_labels_train(arma::colvec &l_train){
  //Same across all derived processing classes
  labels_train=l_train;
}

void Gaussian_smoothing :: set_labels_test(arma::colvec &l_test){
  //Same across all derived processing classes
  labels_test=l_test;
}


// Get data and labels
arma::colvec Gaussian_smoothing :: get_labels_train(){
  return(labels_train);
}

arma::colvec Gaussian_smoothing :: get_labels_test(){
  return(labels_test);
}

vector<arma::mat > Gaussian_smoothing :: get_data_train(){
  return(data_train);
}

vector<arma::mat > Gaussian_smoothing ::get_data_test(){
  return(data_test);
}
