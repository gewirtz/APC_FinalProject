#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <armadillo>

class Optimizer {
 public:
  virtual ~Optimizer() {}
  virtual void fitParams(Model* m);

};

#endif  // OPTIMIZER_H_
