#ifndef JPG_LOAD_LABELS_H_
#define JPG_LOAD_LABELS_H_

#include <armadillo>
#include <string> 

arma::colvec jpg_load_labels(std::string directory, std::string filename); 
//int ppm_load_labels(std::string directory, std::string filename);
#endif