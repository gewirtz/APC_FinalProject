#include <stdio.h>
#include "../processing/mnist_load_images.h"
#include "../processing/mnist_load_labels.h"
#include "../processing/mnist_count_images.h"
#include "../processing/no_processing.h"
#include "../processing/no_processing_test.h"
#include "../processing/gaussian_smoothing.h"
#include "../processing/gs_processing_test.h"
#include "../processing/data_process_base.h"
#include "../ModelFitting/GradientDescent.h"
#include "../ModelFitting/LinearRegression.h"
#include "../ModelFitting/Performance.h"
#include "../ModelFitting/LogisticRegression.h"
#include "../ModelFitting/KNN.h"
#include "../ModelFitting/CrossValidation.h"
#include "../Performance/matplotlibcpp.h"

#include <cmath>
#include <armadillo>
#include <vector>
#include <assert.h>
using namespace arma;
using namespace std;
namespace plt = matplotlibcpp;

namespace{
  void plot_cost(vector<vector<double>> cost, int skip, std::string outfile, std::string model_title){
    int ns = cost.size();
    string title = "Gradient Descent for ";
    title = title.append(model_title);
    title = title.append( " parameter ");

    for(int s=0; s<ns; s++ ){
      int n=cost[s].size();
      std::vector<double> y(n);

      for(int i=0; i < n; i += skip) {
        y.at(i) = cost[s][i];
      } 
      plt::plot(y);
    }
    plt::xlabel("Iteration number");
    plt::ylabel("Cost");
    plt::save(outfile);    
    plt::title(title.append(itoa(s)));
  };



  mat concatenate(vector<arma::mat> input){
    int ex_count = input.size();
    if(ex_count == 0){
      cerr << "Call concatenate on non-empty data " << endl;
      exit(1);
    }
    int num_rows = input[0].n_rows;
    int num_cols = input[0].n_cols;
    mat data = mat(ex_count,num_rows * num_cols); 

    //fill data, rows are examples cols are pixels
    for(int i=0; i<ex_count; i++){
      if(input[i].n_rows!=num_rows || input[i].n_cols!=num_cols ){
        cerr << "Need all input data to have same dimensions\n" << endl;
        exit(-1);
      }
      for(int j=0;j<num_rows;j++){
        for(int k=0;k<num_cols ; k++){
            data(i,j*num_cols+k)=input[i](j,k);
        }
      }
    }
    return(data);
  };
}




int main(int argc, char *argv[]){

  string train_directory, test_directory;
  string train_img, test_img;
  string train_lbl, test_lbl;


  //TO DO: SYNCHRONIZE WITH IMPORT CLASS

  if (argc!=7){
    // keep this for testing
    train_directory = "data/mnist/training/";
    test_directory = "data/mnist/testing/";
    train_lbl = "train-labels.idx1-ubyte";
    train_img = "train-images.idx3-ubyte";
    test_lbl = "t10k-labels.idx1-ubyte";
    test_img = "t10k-images.idx3-ubyte";
  }
  else{

    train_directory = argv[1];
    test_directory =argv[2];
    train_lbl = argv[3];
    train_img = argv[4];
    test_lbl = argv[5];
    test_img = argv[6];
  }


  int unitflag = 0; //change to 1 to run unit testing
  int datatype_flag = 0;  //read last three characters, figure out of .jpg, .
  int process_flag = -1;
  vector<int> model_flag = vector<int>(5);


  //read in preprocessing option 
  while(process_flag != 0 || process_flag != 1 || process_flag != 2 || process_flag != 3){
    cout << endl << "What kind of preprocessing methods should be used?" << endl;
    cout << "Please enter: "  << endl << "0 for just standardization" << endl;
    cout << "1 for Gaussian Smoothing" << endl;
    cout << "2 for Histogram " << endl;
    cout << "3 for both Gaussian Smoothing and Histogram" << endl;
    cin >> process_flag;
    if(process_flag != 0 || process_flag != 1 || process_flag != 2 || process_flag != 3){
      cout << "Please enter a valid selection" << endl;  
    }
  }

  //read in model selection
  cout << endl << "What kind of models would you like to fit?" << endl;
  vector<int> model_flag = vector<int>(5);
  vector<string> candidate_models = vector<string>(5);

  candidate_models[0] = "linear regression"; 
  candidate_models[1] =  "regularized linear regression";
  candidate_models[2] = "logistic regression"; 
  candidate_models[3] =  "regularized logistic regression";
  candidate_models[4] = "k-nearest neighbors"; 


  for(int i = 0; i < model_flag.size();i++){
    model_flag[i] = -1;
    while(model_flag[i] != 0 || model_flag[i] != 1){
      cout << "Would you like to fit " << candidate_models[i] << "?" << endl;
      cout << "Please enter: " << endl;
      cout << "0 if yes" << endl;
      cout << "1 if no" << endl;
      cin >> model_flag[i];
      if(model_flag != 0 || model_flag != 1 ){
        cout << "Please enter a valid selection" << endl;  
      }
    }
  }

  while(model_flag != 0 || model_flag != 1 || model_flag != 2 || model_flag != 3 || model_flag != 4 ){
    cout << endl << "What kind of models would you like to fit?" << endl;
    cout << "Please enter: "  << endl << "0 for linear regression" << endl;
    cout << "1 for regularized linear regression" << endl;
    cout << "2 for logistic regression" << endl;
    cout << "3 for regularized logistic regression  " << endl;
    cout << "4 for k-Nearest Neighbors" << endl;
    cin >> model_flag;
    if(model_flag != 0 || model_flag != 1 || model_flag != 2 || model_flag != 3 || model_flag != 4){
      cout << "Please enter a valid selection" << endl;  
    }
  }



   datatype_flag = atoi( argv[9] );






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
  
  mat c_train_data = concatenate(train_data);
  mat c_tr_data_gauss= concatenate(tr_data_gauss);
  mat c_tt_data = concatenate(tt_data);
  mat c_t_data_gauss = concatenate(t_data_gauss);

  GradientDescent *gd = new GradientDescent(100, .001, 10e-4, 0);
  CrossValidation *cv = new CrossValidation(1.0,21,4,10); 

  cout << "Fitting KNN without preprocessing" << endl;
  KNN *fit_knn0 = new KNN(c_train_data, tr_lbls, cv);
  cout << "Fitting KNN with Gaussian Smoothing" << endl;
  KNN *fit_knn1 = new KNN(c_tr_data_gauss, tr_lbls, cv);
  cout << "Fitting Logisitic Regession without preprocessing" << endl;
  LogisticRegression *fit_lr0 = new LogisticRegression(c_train_data, tr_lbls, gd);
      cout << "Fitting Logistic Regression with Gaussian Smoothing" << endl;
  LogisticRegression *fit_lr1 = new LogisticRegression(c_tr_data_gauss, tr_lbls, gd);


  cout <<"predicting step\n" << endl;
  vector<vec> fits;
  vec v;
  v = fit_knn0->predict(c_tt_data);
  fits.push_back(v);
  v = fit_knn1->predict(c_t_data_gauss);
  fits.push_back(v);
  v = fit_lr0->predict(c_tt_data);
  fits.push_back(v);
  v = fit_lr1->predict(c_t_data_gauss);
  fits.push_back(v); 
  //determines accuracy

  int numClasses = fit_knn0->getLabelSet().size();
  vec pred_lbls;
  string name;
  double num_correct = 0.0;
  vec type1, type2, accByClass, countByClass;
  double max_acc = -1.0;
  string champ = "";

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

    for(int i = 0; i < pred_lbls.size(); i++){
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

    cout << endl << "Results for " << name << ": " << endl;
    cout << endl; 
    for(int i = 0; i < numClasses; i++){
      cout << "For label "<< i << ", the class testing accuracy is " << class_acc[i] << endl;
      cout << "For label " << i << ", the test frequency of type 1 error is " << type1_freq[i] << endl;
      cout << "For label " << i << ", the test frequency of type 2 error is " << type2_freq[i] << endl<<endl;
    }
    cout << endl<< endl;
    cout << "The overall testing accuracy for " << name << " is " <<  total_acc << endl;
    if(total_acc > max_acc){
      max_acc = total_acc;
      champ = name;
    }

  }
  cout << endl << endl;
  cout << "We recommend proceeding with " << champ << endl<< endl; 
}