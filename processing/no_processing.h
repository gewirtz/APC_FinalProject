#ifndef NO_PROCESSING_H_
#define NO_PROCESSING_H_

#include <armadillo>
#include "data_process_base.h"

class No_processing : public Data_processor{

  public:
    No_processing();
    ~No_processing();
    
    void process();
    void set_data_train(arma::mat* imported_data);
    void set_data_test(arma::mat* imported_data);
    void set_labels_train(arma::colvec* imported_labels);
    void set_labels_test(arma::colvec* imported_labels);
  
    arma::colvec* get_labels_train();
    arma::colvec* get_labels_test();
    arma::mat* get_data_train();
    arma::mat* get_data_test()
  
    
  private:
    int has_been_processed; //switch for whether data has been processed
    arma::colvec labels_train;
    arma::colvec labels_test;
    arma::mat data_train; //This will be the processed data for each class
    arma::mat data_test; //This will be the processed data for each class

    
};

#endif //NO_PROCESSING_H_
