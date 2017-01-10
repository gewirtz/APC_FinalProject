#include <stdio.h>
#include "processing/mnist_load_images.h"
#include "processing/mnist_load_labels.h"
#include "processing/mnist_count_images.h"
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
#include <armadillo>
#include <vector>
#include <assert.h>
using namespace arma;
using namespace std;

int main(int argc, char *argv[]){
  /*
  mat A = randu<mat>(4,5);
  vec v = randu<vec>(4);
  srand(1);
  mat B = shuffle(A);
  srand(1);
  vec u = shuffle(v);
  cout << A << endl;
  cout << B << endl;
  cout << v << endl;
  cout << u << endl;
  */

  string train_directory, test_directory;
  string train_img, test_img;
  string train_lbl, test_lbl;
  int unitflag = 0;
  int datatype_flag = 0;
  int process_flag = 0;

  if (argc!=10){
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

    unitflag = atoi( argv[7] );
    process_flag = atoi( argv[8] );
    datatype_flag = atoi( argv[9] );

  }


  vector<arma::mat>  tt_data, train_data;
  arma::colvec train_lbls, test_lbls;

  if (datatype_flag == 0){ // MNIST

    train_data = mnist_load_images(train_directory, train_img, unitflag);
    train_lbls = mnist_load_labels(train_directory, train_lbl);
    tt_data = mnist_load_images(test_directory, test_img, unitflag);
    test_lbls = mnist_load_labels(test_directory, test_lbl);

  }

  else if(datatype_flag == 1){ // PPM

    // train_data = ...

  }

  else if(datatype_flag == 2){ // JPEG

    // train_data = ...

  }
  // size checks
  cout << "train_data size: " << train_data.size() << endl;
  cout << "train_lbl size: " << train_lbls.n_elem << endl;
  cout << "test_data size: " << tt_data.size() << endl;
  cout << "test_lbl size: " << test_lbls.n_elem << endl;

  // step 2: Process the data
          
  No_processing *p_np;
  Gaussian_smoothing *p_gs;
  Histogram *p_hist;
          
  vector<arma::mat> tr_data, t_data;
  arma::colvec tr_lbls, t_lbls;
  vector<arma::mat> tr_data_matform, t_data_matform;

          
  if(process_flag == 0){ // no processing
    p_np=process_driver(train_data,tt_data,train_lbls,test_lbls);
    
    tr_lbls = p_np->get_labels_train();
    t_lbls = p_np->get_labels_test();
    tr_data = p_np->get_data_train();
    t_data = p_np->get_data_test();
    
  }
  else if (process_flag == 1){ // gaussian
    p_gs=process_driver_gs(train_data,tt_data,train_lbls,test_lbls);

    tr_lbls = p_gs->get_labels_train();
    t_lbls = p_gs->get_labels_test();
    tr_data = p_gs->get_data_train();
    t_data = p_gs->get_data_test();

    cout << "gaussian train size: " << tr_data.size() << endl;
    cout << "gassian test size: " << t_data.size() << endl;
  }
  else if (process_flag == 2){ // histogram
    p_hist =  process_driver_hist(train_data, tt_data, train_lbls, test_lbls);
    
    tr_lbls = p_hist->get_labels_train();
    t_lbls = p_hist->get_labels_test();
    tr_data_matform = p_hist->get_data_train();
    t_data_matform = p_hist->get_data_test();

    cout << "number of train histograms: " << tr_data_matform.size() << endl;
    cout << "number of test histograms: " << t_data_matform.size() << endl;
          
    //For histogram implementation change hist matrix to vector of row vectors 
    for(int i=0;i<tr_data_matform[0].n_rows;i++){
      tr_data[i]=tr_data_matform[0].row(i);
    }
    for(int i=0;i<t_data_matform[0].n_rows;i++){
      t_data[i]=t_data_matform[0].row(i);
    }
  }



  // step 3: Model the data
  //cout << "step 3\n" << endl;
  GradientDescent *gd = new GradientDescent(500, .001, 10e-4, 0);
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

  cout << "Gradient Differences " << endl;
  cout << fit->get_exactParams() - fit->get_Params()[0] << endl;

  vector<vector<double>> costs = gd->getLastCost();

  /*TO DO: NOEMI+ANDREAS
  Hi, so i want to make the plot i sent to you on fb 
  xy/2d line graph not scatter
  y-axis vector<double> costs[j] as above, 
  x-axis should be a vector<int> iter_num which contains 1,..., costs[j].size()
  note costs[j].size() == costs[k].size() ie they are all the same length
  title can be Gradient Descent 
  
  x-axis title Iteration number  
  y-axis title Cost 
  legend with parameter j and each line a different color
  esentially this is the plot(x,y) command in python  

  preferrably have the line graphs for =0,1,...,costs.size()-1 all on the same graph 
  but getting one per file is fine if not

  have the system output it and save it to a .png 

  i dont know anything about plotting in c++, you guys are much more
  experienced with it i would assume.

  
  perhaps make this a function in performance or something, along the lines of
  graph(vector<vector<double>> costs, int skip) and have the graph display the cost
  of every skip iterations (ie if skip is 5 show 0,5,10,15,20,...)

  thanks and let me know if you have any other questions!  
  


   THIS DOES NOT WORK
  step 4: output the results - Noemi coded this I think?
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
