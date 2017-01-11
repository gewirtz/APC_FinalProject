#ifndef GAUSSIAN_SMOOTHING_H_
#define GAUSSIAN_SMOOTHING_H_

#include <armadillo>
#include <vector>
#include "data_process_base.h"

class Gaussian_smoothing : public Data_processor{

  public:
  Gaussian_smoothing();
   ~Gaussian_smoothing();
   //Process will be run twice - once for train and once for test
   //Test will always process first
    int process();
    void set_data_train(std::vector<arma::mat > &d_train);
    void set_data_test(std::vector<arma::mat > &d_test);
    void set_labels_train(arma::colvec &l_train);
    void set_labels_test(arma::colvec  &l_test);

    arma::colvec get_labels_train();
    arma::colvec get_labels_test();
    std::vector<arma::mat > get_data_train();
    std::vector<arma::mat > get_data_test();

  private:
    bool train_has_been_processed; //switch for whether data has been processed
    bool test_has_been_processed; //switch for whether data has been processed
    arma::colvec labels_train;
    arma::colvec labels_test;
    std::vector<arma::mat > data_train; //This will be the processed data for each class
    std::vector<arma::mat > data_test; //This will be the processed data for each class


};

#endif //GAUSSIAN_SMOOTHING_H_
