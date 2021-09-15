#ifndef OPT_SGD_MPI_H_
#define OPT_SGD_MPI_H_

#include <iostream>
#include <vector>
#include <mpi.h>
#include "eddl/optimizers/optim.h"
#include "eddl/apis/eddl.h"
#include "mpi_env.hpp"

using namespace std;

typedef Optimizer* optimizer;

class SGD_mpi : public SGD {
public:
    mpi_env* MPE;
    int n_sync;
    int count;
    string mpi_hostname;

    explicit SGD_mpi(mpi_env* MPE, float lr=0.01f, float momentum=0.0f, float weight_decay=0.0f, bool nesterov=false);
    virtual ~SGD_mpi();

    Optimizer *clone() override;
    //Optimizer *share() override;

    void applygrads(int batch) override;
    void sync_grads();
};

optimizer sgd_mpi(mpi_env* MPE, float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false);
#endif
