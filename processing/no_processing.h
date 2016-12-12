#ifndef NO_PROCESSING_H_
#define NO_PROCESSING_H_

#include <armadillo>
#include "data_process_base.h"

class no_processing : public data_processor{

  public:
    no_processing();
    ~no_processing();
    
    arma::mat process();
    void set_Data(arma::mat* imported_data);
    void set_Labels(arma::colvec* imported_labels);
    
  private:
    arma::mat data;
    arma::colvec labels;
    
};

#endif //NO_PROCESSING_H_
