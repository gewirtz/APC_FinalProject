#include <stdio.h>
#include "../processing/mnist_load_images.h"
#include "../processing/mnist_load_labels.h"
#include "../processing/mnist_count_images.h"
#include "../processing/no_processing.h"
#include "../processing/no_processing_test.h"
#include "../processing/gaussian_smoothing.h"
#include "../processing/gs_processing_test.h"
#include "../processing/data_process_base.h"
#include "ModelFitting/GradientDescent.h"
#include "ModelFitting/LinearRegression.h"
#include "ModelFitting/Performance.h"
#include "ModelFitting/LogisticRegression.h"
#include "ModelFitting/KNN.h"
#include "ModelFitting/CrossValidation.h"


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
  
  GradientDescent *gd = new GradientDescent(100, .001, 10e-4, 0);
  CrossValidation *cv = new CrossValidation(1.0,100,20,10); 

  KNN *fit_knn0; = new KNN(train_data, tr_lbls, cv);
  KNN *fit_knn1; = new KNN(tr_data_gauss, tr_lbls, cv);
  LogisticRegression *fit_gd0 = new LogisticRegression(train_data, tr_lbls, cv);
  LogisticRegression *fit_gd1 = new LogisticRegression(tr_data_gauss, tr_lbls, cv);


  cout <<"predicting step\n" << endl;
  vector<vec> fits;
  fits.push_back(fit_knn0->predict(tt_data));
  fits.push_back(fit_knn1->predict(t_data_gauss))
  fits.push_back(fit_gd0->predict(tt_data))
  fits.push_back(fit_gd1->predict(t_data_gauss)) 

  //determines accuracy

  int numClasses = fit->getLabelSet().size();
  vec pred_lbls;
  char* name;
  double num_correct = 0.0;
  vec type1, type2, accByClass, countByClass;

  for(int i = 0; i < fits.size();i++){
    pred_lbls = fits[i];
    if(i == 0){
      name = "KNN without processing";
    }
    else if(i == 1){
      name = "KNN with Gaussian Processing";
    }
    else if(i == 2){
      name = "LogisticRegression without Processing";

    }
    else{
      name = "Logistic Regression with Gaussian Processing";
    }

    num_correct = 0.0;
    type1 = type1.zeros(numClasses); //saying it is class i but it isnt ie s
    type2 = type2.zeros(numClasses);  //predicting it is not class i but it is 
    accByClass = accByClass.zeros(numClasses);
    countByClass = countByClass.zeros(numClasses);
    bool correct;

    for(int i = 0; i < numClasses; i++){
      correct = false;
      countByClass[test_lbls(i)] += 1.0;
      
      if(pred_lbls(i) == test_lbls[i]){
        num_correct += 1.0;
        correct = true;
      }

      if(correct){
        accByClass[test_lbls(i)] += 1.0;
      }
      else{
        type2[test_lbls(i)] += 1.0;
        type1[pred_lbls(i)] += 1.0;
      }
    }


    double total_acc = num_correct / pred_lbls.size();
    vec type1_freq = type1 / (pred_lbls.size() - countByClass);
    vec class_acc = accByClass / countByClass; 
    vec type2_freq = type2/countByClass;   
    //accuracy by class

    cout << "Results for " << name << ": " << endl;
    cout << endl; 
    for(int i = 0; i < fit->getLabelSet().size(); i++){
      cout << "For label "<< i << ", the class testing accuracy is " << class_acc[i] << endl;
      cout << "For label " << i << ", the test frequency of type 1 error is " << type1_freq[i] << endl;
      cout << "For label " << i << ", the test frequency of type 2 error is " << type2_freq[i] << endl;
      cout << endl;
    }
    cout << endl;
    cout << endl;
    cout << "The overall testing accuracy for " << name << " is " <<  total_acc << endl;
  }
}
