#include <iostream>
#include <fstream>
#include <armadillo>
#include <cassert>
#include "ppm_load_labels.h"

using namespace std;
using namespace arma; 

arma::colvec ppm_load_labels(string directory, string filename){
//int ppm_load_labels(string directory, string filename){

  string fname = directory + filename;

  ifstream ppm_label (fname.c_str());
  if(!ppm_label){
  	cout << "Unable to open labelfile" <<endl;
  }

  //Count number of lines
  string line;
  int lines_count = 0;
  while(getline(ppm_label,line)){
  	++lines_count;
  }

  arma::colvec labels = arma::zeros<arma::colvec>(lines_count);

  // read the labels
  for(int i = 0; i < lines_count; i++){
    unsigned char cur_label = 0;
    ppm_label.read( (char*) &cur_label, sizeof(cur_label) );
    labels(i) = (double) cur_label;
  }

  return labels;
}

// For testing purposes
// int main(){

// 	arma::colvec num = ppm_load_labels("/home/andreas/APC524/Project/data/cars/","testing_labels_cars");
// 	cout << num <<endl;
// }