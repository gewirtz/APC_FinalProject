#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "HyperModel.h"
#include "model.h"
class Model;
class HyperModel;

class Optimizer {
 public:
  virtual ~Optimizer() {}
  virtual void fitParams(Model *m) = 0;
  virtual void fitParams(HyperModel *m) = 0;

};

#endif  // OPTIMIZER_H_
