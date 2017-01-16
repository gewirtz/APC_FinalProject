#include <iostream>
#include <fstream>
#include <armadillo>
#include <cassert>
#include "jpg_load_labels.h"

using namespace std;
using namespace arma; 

arma::colvec jpg_load_labels(string directory, string filename){
//int ppm_load_labels(string directory, string filename){

  string fname = directory + filename;

  ifstream jpg_label (fname.c_str());
  if(!jpg_label){
  	cout << "Unable to open labelfile" <<endl;
  }

  //Count number of lines
  string line;
  int lines_count = 0;
  while(getline(jpg_label,line)){
  	++lines_count;
  }

  arma::colvec labels = arma::zeros<arma::colvec>(lines_count);

  // read the labels
  for(int i = 0; i < lines_count; i++){
    unsigned char cur_label = 0;
    jpg_label.read( (char*) &cur_label, sizeof(cur_label) );
    labels(i) = (double) cur_label;
  }

  return labels;
}

// For testing purposes
// int main(){

// 	arma::colvec num = ppm_load_labels("/home/andreas/APC524/Project/data/cars/","testing_labels_cars");
// 	cout << num <<endl;
// }