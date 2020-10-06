#ifndef HE_INIT_H_
#define HE_INIT_H_

#include "eddl/initializers/initializer.h"
#include "eddl/apis/eddl.h"
typedef Layer* layer;

class IHeNormal : public Initializer {
public:
    int seed;

    explicit IHeNormal(int seed=-1);
    void apply(Tensor *params) override;
};

layer he_normal(layer l, int seed=1234);
#endif
