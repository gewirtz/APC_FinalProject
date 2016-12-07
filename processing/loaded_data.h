#ifndef LOADED_DATA_H
#define LOADED_DATA_H


#include <string.h>
#include <iostream>
#include <armadillo>


struct loaded_data{

  std::string train_directory, test_directory,
    train_lbl, train_img,
    test_lbl, test_img;

  std::vector<arma::mat> all_images;
  arma::colvec labels;

  int num_img;
};

#endif
