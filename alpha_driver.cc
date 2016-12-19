#include <stdio.h>
#include "processing/mnist_load_images.h"
#include "processing/mnist_load_labels.h"
#include "processing/mnist_count_images.h"
#include "processing/no_processing.h"
#include "processing/no_processing_test.h"
#include "processing/data_process_base.h"
#include "ModelFitting/GradientDescent.h"
#include "ModelFitting/LinearRegression.h"
#include "ModelFitting/Performance.h"
#include <armadillo>
#include <vector>

using namespace arma;
using namespace std;

int main(){
  

  // step 1: load the data
     //if(argc !=6){
   //printf("Missing inputs, Usage: train directory, test directory, train label file name, train image file name, t   est label file name, test label file name")
   //exit(1);
  //}
   
  string train_directory = "data/mnist/training/";
  //string train_directory =argv[1];
  
  string test_directory = "data/mnist/testing/";
  //string test_directory =argv[2];
  
  string train_lbl = "train-labels.idx1-ubyte";
  //string train_lbl = argv[3];
  
  string train_img = "train-images.idx3-ubyte";
  //string train_img = argv[4];
  
  string test_lbl = "t10k-labels.idx1-ubyte";
  //string test_lbl = argv[5];
  
  string test_img = "t10k-images.idx3-ubyte";
  //string test_img = argv[6];
  
  
  vector<arma::mat>  tt_data, train_data;
  arma::colvec train_lbls, test_lbls;
  
  train_data = mnist_load_images(train_directory, train_img);
  train_lbls = mnist_load_labels(train_directory, train_lbl);
  tt_data = mnist_load_images(test_directory, test_img);
  test_lbls = mnist_load_labels(test_directory, test_lbl);

  // step 2: Process the data
  //cout << "step 2\n" << endl;
  No_processing *p_np;
  p_np=process_driver(train_data,tt_data,train_lbls,test_lbls);


  // Hi Chase/Ari, here is how you would get the labels.
  arma::colvec tr_lbls, t_lbls;
  tr_lbls = p_np->get_labels_train();
  t_lbls = p_np->get_labels_test();
  vector<arma::mat> tr_data, t_data;
  tr_data = p_np->get_data_train();
  t_data = p_np->get_data_test();



  // assert check for out of bounds calls
  //cout << (*temp)(0) << endl;
  //cout << (*temp) << endl;

  
  // step 3: Model the data
  //cout << "step 3\n" << endl;
  GradientDescent *gd = new GradientDescent(1000, .001, .00001);
  LinearRegression *fit = new LinearRegression(tr_data, tr_lbls, gd);
 
  cout <<"predicting step\n" << endl;
  arma::vec pred_lbls = fit->predict(t_data);
  
  double correct = 0.0;
  for(int i = 0; i < pred_lbls.size(); i++){
    if(pred_lbls(i) == test_lbls[i]){
      correct += 1.0;
    }
  }
  double stat3 = correct / pred_lbls.size();

  //cout << "Actual Gradient Values " << endl;
  //cout << fit->get_exactParams() << endl;

  /* THIS DOES NOT WORK
  // step 4: output the results - Noemi coded this I think?
  Performance Pf;
  vec stat1, stat3;
  mat stat2;
  stat1 = Pf.mse(pred_lbls,test_lbls,pred_lbls.size());
  stat2 = Pf.correl(pred_lbls,test_lbls);
  stat3 = Pf.accuracy(pred_lbls,test_lbls);
    */
  cout << "The testing accuracy is " <<  stat3 << endl;

  return 0;
}
