#include "data_process_base.h"
#include <stdio.h>

class no_processing : private data_processor{
  public:
    no_processing(){
      //Same across all derived processing classes
      data_train=new arma::mat; //Check w Jeffrey - or should we just call Mat() constructor here?
      data_test=new arma::mat;
      labels_train=new arma::colvec; //Check w Jeffrey
      labels_test=new arma::colvec;
      has_been_processed=0;
    };
    ~no_processing(){
      //Same across all derived processing classes
      delete data_train;
      delete data_test;
      delete labels_train;
      delete labels_test;
    }

    int process(){
      //Other versions of this class will have Gaussian stuff implemented here, etc.
      //This is the meat of the processing implementation work
      if(has_been_processed==1){
        //Don't let user process already processed data but also no need to throw up lots of errors
        return(1);
      };
      has_been_processed=1;
      return(1);
    };

    // Pass the data and the labels
    void set_data_train(arma::mat* d_train){
      //Same across all derived processing classes
      data_train=d_train;

    };
  
     void set_data_test(arma::mat* d_test){
      //Same across all derived processing classes
      data_test=d_test;

    };
  
      void set_labels_train(arma::mat* l_train){
      //Same across all derived processing classes
      labels_train=l_train;

    };
  
     void set_labels_test(arma::mat* l_test){
      //Same across all derived processing classes
      labels_test=l_test;

    };
  
  
    // Get data and labels
    arma::colvec* get_labels_train(){
      return(labels_train);
    };
  
      arma::colvec* get_labels_test(){
      return(labels_test);
    }
    
    arma::mat* get_data_train(){
      return(data_train);
    }; 
    arma::mat* get_data_test(){
      return(data_test);
    };

  private:
    int has_been_processed; //switch for whether data has been processed
    arma::colvec labels_train;
    arma::colvec labels_test;
    arma::mat data_train; //This will be the processed data for each class
    arma::mat data_test; //This will be the processed data for each class


};
