#ifndef NO_PROCESSING_H_
#define NO_PROCESSING_H_

#include <armadillo>
#include <vector>
#include "data_process_base.h"

class No_processing : public Data_processor{

  public:
    No_processing();
   ~No_processing();
    
    void process();
    void set_data_train(std::vector<arma::mat > &d_train);
    void set_data_test(std::vector<arma::mat > &d_test);
    void set_labels_train(arma::colvec &l_train);
    void set_labels_test(arma::colvec  &l_test);
  
    arma::colvec* get_labels_train();
    arma::colvec* get_labels_test();
    std::vector<arma::mat >* get_data_train();
    std::vector<arma::mat >* get_data_test();
      
  private:
    int has_been_processed; //switch for whether data has been processed
    arma::colvec labels_train;
    arma::colvec labels_test;
    std::vector<arma::mat > data_train; //This will be the processed data for each class
    std::vector<arma::mat > data_test; //This will be the processed data for each class

    
};

#endif //NO_PROCESSING_H_
