#ifndef MPI_ENV_H_
#define MPI_ENV_H_

#include <mpi.h>
#include "eddl/apis/eddl.h"
#include "eddl/net/net.h"

class mpi_env {
public:
    int n_sync;
    int bl;
    size_t mpi_block;
    float div;
    int mpi_size;
    int mpi_rank;
    float *avg_data;

    mpi_env(int n_sync=1, int bl=1024);
    ~mpi_env();
    void Barrier();
    void Bcast_Tensor(Tensor* t_in, int root);
    float Gather_and_average(float send_data);
    void Allreduce_Tensor(Tensor* t_in);
    void Broadcast_params(Net* net);
};

#endif

