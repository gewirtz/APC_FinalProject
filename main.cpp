#include <stdio.h>
#include "processing/mnist_load_images.h"
#include "processing/mnist_load_labels.h"
#include "processing/mnist_count_images.h"
#include "processing/mnist_count_images.h"
//#include "processing/ppm_load_images.h"
//#include "processing/ppm_load_labels.h"
//#include "processing/jpg_load_images.h"
//#include "processing/jpg_load_labels.h"
#include "processing/no_processing.h"
#include "processing/no_processing_test.h"
#include "processing/gaussian_smoothing.h"
#include "processing/gs_processing_test.h"
#include "processing/histogram.h"
#include "processing/histogram_test.h"

#include "processing/data_process_base.h"
#include "ModelFitting/GradientDescent.h"
#include "ModelFitting/LinearRegression.h"
#include "ModelFitting/Performance.h"
#include "ModelFitting/LogisticRegression.h"
#include "ModelFitting/KNN.h"
#include "ModelFitting/CrossValidation.h"
#include "Performance/matplotlibcpp.h"

#include <cmath>
#include <armadillo>
#include <vector>
#include <assert.h>

using namespace arma;
using namespace std;
namespace plt = matplotlibcpp;

namespace{
  //void plot_cost(vector<vector<double>> cost, int skip, std::string outfile, std::string model_title){
 void plot_cost(vector<vector<double>> cost, std::string outfile, std::string model_title){
    int ns = cost.size();
    string title = "Gradient Descent for ";
    title.append(model_title);
    title.append( " parameter ");
    string temp;

    for(int s=0; s<ns; s++ ){
      int n=cost[s].size();
      std::vector<double> y(n);

      //for(int i=0; i < n; i += skip) {
      for(int i=0; i < n; i++) {
        y.at(i) = cost[s][i];
      } 
      plt::plot(y);
      
      plt::xlabel("Iteration number");
      plt::ylabel("Cost");
      temp = title;
      plt::title(temp.append(to_string(s)));
      temp = outfile.append("_forParam_");
      cout << "saving" << endl;
      plt::save(temp.append(to_string(s)));  //will segfault if there is already w/ same filename in directory
    }
  };


//helper function to concatenate train and test data into a form usable by model object
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
  vector<arma::mat>  tt_data, train_data;
  arma::colvec train_lbls, test_lbls;

  int unitflag = 0; //change to 1 to run unit testing
  int datatype_flag = 0;  //TO DO: from passed path, read last three characters, figure out if .jpg, .ppm, or mnist




  if (argc!=7){
    // keep this for testing
    train_directory = "data/mnist/training/";
    test_directory = "data/mnist/testing/";
    train_lbl = "train-labels.idx1-ubyte";
    train_img = "train-images.idx3-ubyte";
    test_lbl = "t10k-labels.idx1-ubyte";
    test_img = "t10k-images.idx3-ubyte";
  }
/* 

TODO : ANDREAS - 

I cannot seem to get it to compile with jpg or ppm included, please fix so after uncommented the
else statement compiles

  else{

    train_directory = argv[1];
    test_directory =argv[2];
    train_lbl = argv[3];
    train_img = argv[4];
    test_lbl = argv[5];
    test_img = argv[6];
    string suffix = argv[4].find_last_of(3);
  if(suffix == ".jpg" || argv[4].find_last_of(3) == ".jpeg"){
      train_data = jpg_load_images(train_directory, train_img, unitflag);
      train_lbls = jpg_load_labels(train_directory, train_lbl);
      tt_data = jpg_load_images(test_directory, test_img, unitflag);
      test_lbls = jpg_load_labels(test_directory, test_lbl);

    }
    else if( suffix == ".ppm"){
      train_data = ppm_load_images(train_directory, train_img, unitflag);
      train_lbls = ppm_load_labels(train_directory, train_lbl);
      tt_data = ppm_load_images(test_directory, test_img, unitflag);
      test_lbls = ppm_load_labels(test_directory, test_lbl);
    }

    else{*/
      train_data = mnist_load_images(train_directory, train_img, unitflag);
      train_lbls = mnist_load_labels(train_directory, train_lbl);
      tt_data = mnist_load_images(test_directory, test_img, unitflag);
      test_lbls = mnist_load_labels(test_directory, test_lbl);
      /*
    }
  }
*/


/* //////////////////////////////////// End-user selections ////////////////////////////////////////////////// */
  vector<int> process_flag = vector<int>(3);
  vector<string> process_names = vector<string>(3);
  process_names[0] = "standardization"; 
  process_names[1] =  "gaussian smoothing";
  process_names[2] = "histogram"; 

  for(int i = 0; i < process_flag.size();i++){
    process_flag[i] = -1;
    while(process_flag[i] != 0 && process_flag[i] != 1){
      cout << "Would you like to use " << process_names[i] << " to preprocess the data?" << endl;
      cout << "Please enter: " << endl;
      cout << "0 if yes" << endl;
      cout << "1 if no" << endl;
      cin >> process_flag[i];
      if(process_flag[i] != 0 && process_flag[i] != 1 ){
        cout << "Please enter a valid selection" << endl;  
      }
    }
  }

  //read in model selection
  cout << endl << "Model selection decision:" << endl << endl;
  vector<int> model_flag = vector<int>(5);
  vector<string> model_names = vector<string>(5);

  model_names[0] = "linear regression"; 
  model_names[1] =  "regularized linear regression";
  model_names[2] = "logistic regression"; 
  model_names[3] =  "regularized logistic regression";
  model_names[4] = "k-nearest neighbors"; 


  for(int i = 0; i < model_flag.size();i++){
    model_flag[i] = -1;
    while(model_flag[i] != 0 && model_flag[i] != 1){
      cout << "Would you like to fit " << model_names[i] << "?" << endl;
      cout << "Please enter: " << endl;
      cout << "0 if yes" << endl;
      cout << "1 if no" << endl;
      cin >> model_flag[i];
      if(model_flag[i] != 0 && model_flag[i] != 1 ){
        cout << "Please enter a valid selection" << endl;  
      }
    }
  }


  int num_iter, batchSize, num_folds;
  double tol, learnRate;

  //create gradient descent object
    GradientDescent *gd; 
    if(model_flag[0] == 0 || model_flag[1] == 0 || model_flag[2] == 0 || model_flag[3] == 0){
      num_iter = -1;
      while(num_iter <= 0){
        cout << endl << "For how many iterations do you want to run gradient descent?" << endl;
        cout << "For speed choose ~100, for optimal fit choose ~10,000" << endl; //MAKE SURE THIS IS CORRECT
        cin >> num_iter;
        if(num_iter <= 0){
          cout << "Need positive number of iterations" << endl;
        }
      }
      batchSize = -1;
      while(batchSize < 0){
        cout << endl << "How large do you want the batches to be in the gradient descent algorithm?" << endl;
        cout << "Enter 0 for batch gradient descent" << endl;
        cin >> batchSize;
        if(batchSize < 0){
          cout << "Need nonegative batch size" << endl;
        }
      }
      
      tol = -1;
      while(tol <= 0){
        cout << endl << "What is the threshold for convergence of gradient descent? (eg 10e-3)" << endl;
        cin >> tol;
        if(tol <= 0){
          cout << "Need positive tolerence" << endl;
        }
      }
      

      learnRate = -1;
      while(learnRate <= 0){
        cout << endl << "What do you want the initial learning rate for gradient descent? (eg 10e-2)"  << endl;
        cin >> learnRate;
        if(learnRate <= 0){
          cout << "Need positive learning rate" << endl;
        }
      }

      gd = new GradientDescent(num_iter, learnRate, tol, batchSize);
    }


//create cross_validation object
    CrossValidation *cv;
    if(model_flag[1] == 0 || model_flag[3] == 0 || model_flag[4] == 0 ){
      num_folds = -1;
      while(num_folds <= 0){
        cout << endl << "How many folds would you like to use for cross validation? (eg 4)"  << endl;
        cin >> num_folds ;
        if(num_folds  <= 0){
          cout << "Need positive number of folds" << endl;
        }
      }
      cv = new CrossValidation(1.0,21,4,num_folds); //TO DO ARI: SHOULD OTHER ARGUMENTS BE USER INPUT?
    }


/* //////////////////////////////////// Preprocessing Data ////////////////////////////////////////////////// */
  vector<mat> processed_tr_data;
  vector<mat> processed_t_data;


  vector<mat> p_train; 
  vector<mat> p_test;

  cout << "Preprocessing data" << endl << endl;
  vector<string> used_prep;  


  for(int i = 0; i < process_flag.size(); i++){
    if(process_flag[i] == 1){
      continue;
    }
    //user wants to include preprocessing i
    cout << "Implementing " << process_names[i] << endl;
    used_prep.push_back(process_names[i]);

    if(i == 0){ //no process
      No_processing *p_np;
      p_np = process_driver(train_data,tt_data,train_lbls,test_lbls);
      p_train = p_np->get_data_train() ;
      p_test = p_np->get_data_test(); 
      processed_tr_data.push_back(concatenate(p_train));
      processed_t_data.push_back(concatenate(p_test));
    }

    else if(i == 1){ //gaussian smoothing
      Gaussian_smoothing *p_gs;
      p_gs = process_driver_gs(train_data,tt_data,train_lbls,test_lbls);
      p_train = p_gs->get_data_train();
      p_test = p_gs->get_data_test();
      processed_tr_data.push_back(concatenate(p_train));
      processed_t_data.push_back(concatenate(p_test));
    }

    else if(i == 2){ 
      Histogram *p_hist;
      p_hist = process_driver_hist(train_data,tt_data,train_lbls,test_lbls);
      vector<mat> p_train_temp = p_hist->get_data_train();
      vector<mat> p_test_temp = p_hist->get_data_test();
     //For histogram implementation change hist matrix to vector of row vectors 
      for(int i=0;i<p_train_temp[0].n_rows;i++){
        p_train[i]=p_train_temp[0].row(i);
      }
      for(int i=0;i<p_test_temp[0].n_rows;i++){
        p_test[i]=p_test_temp[0].row(i);
      }
      processed_tr_data.push_back(concatenate(p_train));
      processed_t_data.push_back(concatenate(p_test));
    }
  }



/* /////////////////////////MODEL FITTING /////////////////////////////////////// */

  cout << endl << "Model fitting" << endl;
  vector<vec> fits;
  vector<string> fit_names;
  vec fit;
  string name;
  string s; 

  LinearRegression *linr;
  LogisticRegression *logr;
  KNN *knn;
  int numClasses;

  for(int i = 0; i < model_flag.size();i++){
      if(model_flag[i] == 1){
        continue;
      }
   for(int j = 0; j < processed_tr_data.size(); j++){
      name = model_names[i];
      name = name.append(" on ");
      name = name.append(used_prep[j]);
      name = name.append(" data");
      fit_names.push_back(name);

      cout << "Fitting " << name << endl;
     
      if(i == 0){
        linr = new LinearRegression(processed_tr_data[j],train_lbls,gd);
        fit = linr->predict(processed_t_data[j]);
      //show gradient descent paths 
        s = "GradDesc_LinReg_";
        plot_cost(gd->getLastCost(), s.append(used_prep[j]), name);
        fits.push_back(fit);
        numClasses = linr->getLabelSet().size();
      }
      else if(i == 1){
        //TO DO IMPLEMENT REGULARIZED LINEAR REGRESSION
      }
      else if(i == 2){
        logr = new LogisticRegression(processed_tr_data[j],train_lbls,gd);
        fit = logr->predict(processed_t_data[j]);
        fits.push_back(fit);
        //show gradient descent paths 
        s = "GradDesc_LogReg_";
        cout << "About to plot" << endl;
        plot_cost(gd->getLastCost(), s.append(used_prep[j]), name);
        numClasses = logr->getLabelSet().size();
      }
      else if(i == 3){
        //TO DO IMPLEMENT REGULARIZED LOGISTIC REGRESSION
      }
      else if(i == 4){
        knn = new KNN(processed_tr_data[j], train_lbls, cv);
        fit = knn->predict(processed_t_data[j]);
        fits.push_back(fit);
        numClasses = knn->getLabelSet().size();
      }
    }
  }  

/* ///////////////////////////////////////// Diagnostics /////////////////////////////////////////////////////// */



  vec pred_lbls;

  double num_correct;
  vec type1, type2, accByClass, countByClass;
  double max_acc = -1.0;
  string champ = "";

  for(int i = 0; i < fits.size();i++){
    pred_lbls = fits[i];
    name = fit_names[i];

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





  //free allocated memory
  bool del_cv = false;
  bool del_gd = false;

  for(int i = 0; i < model_flag.size();i++){
    if(model_flag[i] == 1){
          continue;
    }
    if(i == 0){
      delete(linr);
      del_gd = true;
    }
    if(i == 1){
      //IMPLEMENT reg linr
    }
    if(i == 2){
      //IMPLEMENT gd
    }
    if(i == 3){
      delete(logr);
      del_gd = true;
    }
    if(i == 4){
      delete(knn);
      del_cv = true;
    }
  }
  if(del_cv){
    delete(cv);
  }
  if(del_gd){
    delete(gd);
  }


  return(0);
}