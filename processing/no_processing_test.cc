#include <stdio.h>
#include "no_processing.h"

No_processing* main(arma::mat* imported_data, arma::colvec* imported_labels){

  No_processing p;
  p = new No_processing();
  p->set_Data(imported_data);
  p->set_Labels(imported_labels);
  p->process // process is in the NO_PROCESSING_H_ header definition
  return(p*);
 };
