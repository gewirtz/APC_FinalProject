#include <string.h>
#include <vector>
#include <armadillo>

#include "mnist_load_images.cc"
#include "mnist_load_labels.cc"

#invlude "data_load_mnist.h"

using namespace arma;

mnist_load_driver :: data_load_mnist(string directory, string lbl_fname, string img_fname)
	: directory_(directory),
	  lbl_fname_(lbl_fname),
	  img_fname_(img_fname),
{}

mnist_load_driver :: ~data_load_mnist()
{}

arma::mat data_load_mnist::process(string directory, string img_fname){
  vector<arma::mat> all_images;
  all_images = mnist_load_images(directory_, img_fname_, all_images);
  
  cout << all_images.size() << endl; // how many images you have
  cout << all_images[0].size() << endl; // how big each image is
  
  return all_images;
}

arma::colvec data_load_mnist::data_labels(string directory, string lbl_fname){

  arma::colvec labels;
  labels = mnist_load_labels(directory_, lbl_fname_);

  return labels;

}

/*
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
  

  
  // take a look at an image
  cout << all_images[0] << endl;

  cout << labels.size() << endl; // how many labels you have
  cout << labels[0] << endl; // look at a label
  
}
*/
