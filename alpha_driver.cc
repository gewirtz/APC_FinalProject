#include <stdio.h>
#include "mnist_load_images.cc"
#include "mnist_load_labels.cc"


void main(int argc, char* argv[]){
   //if(argc !=6){
   //printf("Missing inputs, Usage: train directory, test directory, train label file name, train image file name, test label file name, test label file name")
   //exit(1);
   }
   
  string train_directory = "../data/mnist/training/";
  //string train_directory =argv[1];
  
  string test_directory = "../data/mnist/testing/";
  //string test_directory =argv[2];
  
  string train_lbl = "train-labels.idx1-ubyte";
  //string train_lbl = argv[3];
  
  string train_img = "train-images.idx3-ubyte";
  //string train_img = argv[4];
  
  string test_lbl = "t10k-labels.idx1-ubyte";
  //string test_lbl = argv[5];
  
  string test_img = "t10k-images.idx3-ubyte";
  //string test_img = argv[6];
  
  arma::mat train_data, test_data;
  arma::colvec train_lbls, test_lbls;
   
  train_data = mnist_load_images(train_directory, train_img);
  train_lbls = mnist_load_labels(train_directory, train_lbl);
  test_data = mnist_load_images(test_directory, test_img);
  test_lbls = mnist_load_labels(test_directory, test_lbl);
  
  //Begin model portion
};
