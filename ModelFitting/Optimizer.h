#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

class Model;
class KNN;
class GradientModel;


class Optimizer {
 public:
  virtual ~Optimizer() {}
  virtual void fitParams(Model *m) = 0;
  virtual void fitParams(KNN *m) = 0;
  virtual void fitParams(GradientModel *m) = 0;

};

#endif  // OPTIMIZER_H_
