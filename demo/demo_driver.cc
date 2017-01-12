#include <stdio.h>
#include "../processing/mnist_load_images.h"
#include "../processing/mnist_load_labels.h"
#include "../processing/mnist_count_images.h"
#include "../processing/no_processing.h"
#include "../processing/no_processing_test.h"
#include "../processing/gaussian_smoothing.h"
#include "../processing/gs_processing_test.h"
#include "../processing/data_process_base.h"
#include <armadillo>
#include <vector>
#include <assert.h>
using namespace arma;
using namespace std;

int main(int argc, char *argv[]){

  string train_directory, test_directory;
  string train_img, test_img;
  string train_lbl, test_lbl;
  int unitflag = 0;
  int datatype_flag = 0;
  int process_flag = 0;
  
  train_directory = "../data/mnist/training/";
  test_directory = "../data/mnist/testing/";
  train_lbl = "train-labels.idx1-ubyte";
  train_img = "train-images.idx3-ubyte";
  test_lbl = "t10k-labels.idx1-ubyte"; 
  test_img = "t10k-images.idx3-ubyte";

  vector<arma::mat> train_data, tt_data, tr_data_gauss, t_data_gauss;
  arma::colvec train_lbls,test_lbls, tr_lbls, t_lbls;

  train_data = mnist_load_images(train_directory, train_img, unitflag);
  train_lbls = mnist_load_labels(train_directory, train_lbl);
  tt_data = mnist_load_images(test_directory, test_img, unitflag);
  test_lbls = mnist_load_labels(test_directory, test_lbl);

  Gaussian_smoothing *p_gs;
  assert(p_gs != NULL);
 
  p_gs = process_driver_gs(train_data,tt_data,train_lbls,test_lbls);

  tr_lbls = p_gs->get_labels_train();
  t_lbls  = p_gs->get_labels_test();
  tr_data_gauss = p_gs->get_data_train();
  t_data_gauss = p_gs->get_data_test();

  vector<vector<arma::mat>> processed_data_train(2), 
                            processed_data_test(2);

  processed_data_train[0] = train_data;
  processed_data_train[1] = tr_data_gauss;
  processed_data_test[0]  = tt_data;
  processed_data_test[1]  = t_data_gauss;
 
  // modelling people have fun!!! I'm SO HUNGRY!!!!
  // - Nina & Bill
}
