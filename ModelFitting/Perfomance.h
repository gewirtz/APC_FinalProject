#ifndef PERFORMANCE_H_
#define PERFORMANCE_H_



class Performance : public Model  {
 public:

  arma::vec mse(double *label, double *predict_label); //gives exact solution 



 };

#endif  // PERFORMANCE_H_
