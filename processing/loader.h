#ifndef LOADER_H
#define LOADER_H

#include <armadillo>
class loader
{

  string train_directory, test directory,
    train_img, train_lbl,
    test_img, test_lbl;

  int num_img;

  vector<arma::mat> all_images; 
  arma::colvec labels;

 public:
  mnist_load_images(string directory, string filename,
		    vector<arma::mat> &all_images);

  mnist_load_labels(string directory, string filename,
		    arma::colvec labels)

#endif
