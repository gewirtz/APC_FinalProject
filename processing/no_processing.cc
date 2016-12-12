#include "data_process_base.h"
#include <stdio.h>

class no_processing : private data_processor{
  public:
    no_processing(){
      //Same across all derived processing classes
      data=new arma::mat; //Check w Jeffrey - or should we just call Mat() constructor here?
      labels=new arma::colvec; //Check w Jeffrey
    };
    ~no_processing(){
      //Same across all derived processing classes
      delete data;
      delete labels;
    }

    arma::mat process(){
      //Other versions of this class will have Gaussian stuff implemented here, etc.
      //This is the meat of the processing implementation work
      return data;
    };

    void set_Data(arma::mat* imported_data){
      //Same across all derived processing classes
      data=imported_data;

    };

    void set_Labels(arma::colvec* imported_labels){
      //Same across all derived processing classes
      labels=imported_labels;
    };

  private:
    arma::colvec labels;
    arma::mat data; //This will be the processed data for each class 

};
