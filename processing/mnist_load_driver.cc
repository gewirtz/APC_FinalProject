#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <assert.h>
#include <armadillo>

#include "reverse.cc"
#include "mnist_load_images.cc"
#include "mnist_load_labels.cc"
#include "mnist_count_images.cc"

#invlude "data_load_mnist.h"

using namespace arma;

mnist_load_driver :: data_load_mnist(string directory, string lbl_fname, string img_fname)
	: directory_(directory),
	  lbl_fname_(lbl_fname),
	  img_fname_(img_fname),
{}

mnist_load_driver :: ~data_load_mnist()
{}

// don't need to count images any more I think
/*
int data_load_mnist :: num_img(string directory, string img_fname){

int num_imgs;
num_imgs = mnist_count_images(directory_, img_fname_);
return num_imgs;
} 
*/

arma::mat data_load_mnist::process(string directory, string img_fname){
  vector<arma::mat> all_images;
  all_images = mnist_load_images(directory_, img_fname_, all_images);
  
  return all_images;
}

arma::colvec data_load_mnist::data_labels(string directory, string lbl_fname){

arma::colvec labels; //= arma::zeros<arma::colvec>(num_img);

labels = mnist_load_labels(directory_, lbl_fname_);

return labels;

}


/*
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

  // load the images
  //vector<vector<double> > all_images;
  vector<arma::mat> all_images;
  mnist_load_images(train_directory, train_img, all_images);

  // return the number of images
  int num_img;
  num_img = mnist_count_images(train_directory,train_img, all_images);

  // load the labels
  //vector<double> labels(num_img);

  arma::colvec labels = arma::zeros<arma::colvec>(num_img);
  mnist_load_labels(train_directory, train_lbl, labels);


  // check to make sure you did it right
  
  cout << all_images.size() << endl; // how many images you have
  cout << all_images[0].size() << endl; // how big each image is
  
  // take a look at an image
  cout << all_images[0] << endl;

  cout << labels.size() << endl; // how many labels you have
  cout << labels[0] << endl; // look at a label
  
}
*/