#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <assert.h>

#include "reverse.cc"
#include "mnist_load_images.cc"
#include "mnist_load_labels.cc"

// at the request of the modeling team...
//#include <armadillo>

int main(){

  // these will likely be taken from config file
  string train_directory = "../data/mnist/training/";
  string test_directory = "../data/mnist/testing/";

  string train_lbl = "train-labels.idx1-ubyte";
  string train_img = "train-images.idx3-ubyte";
  string test_lbl = "t10k-labels.idx1-ubyte";
  string test_img = "t10k-images.idx3-ubyte";


  // we are going to put things into an ARMADILLO MATRIX
  // which is really just a vector
  // vector<arma::mat> all_images;

  // DOUBLE VECTOR Implementation

  // load the images
  vector<vector<double> > all_images;
  mnist_load_images(train_directory, train_img, all_images);

  // return the number of images
  int num_img;
  num_img = mnist_count_images(train_directory_train_img, all_images);

  // load the labels
  vector<double> labels(num_img);
  mnist_load_labels(train_directory, train_img, labels);


  // check to make sure you did it right
  
  cout << all_images.size() << endl; // how many images you have
  cout << all_images[0].size() << endl; // how big each image is
  
  // take a look at an image
  for(int i = 0; i < all_images[0].size(); i++){
    cout << all_images[0][i] << " ";

    if (i % 28 == 0){
      cout << "\n";
    }
  }

  cout << "\n";
  
  cout << labels.size() << endl; // how many labels you have
  cout << labels[0] << endl; // look at a label
}
