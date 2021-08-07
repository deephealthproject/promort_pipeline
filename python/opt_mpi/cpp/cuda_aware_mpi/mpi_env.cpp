#include "mpi_env.hpp"

mpi_env::mpi_env(int n_sync, int bl):n_sync(n_sync), bl(bl){
    int h_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Get_processor_name(hostname, &h_len);
    
    mpi_block = bl * 1024;
    div = 1/(static_cast<float>(mpi_size)); 

    avg_data = new float [mpi_size];
    std::cout << "MPI_ENV Constructor, hello from " << hostname << ", rank " << mpi_rank << std::endl;
}

mpi_env::~mpi_env(){delete[] avg_data; MPI_Finalize();}

void mpi_env::Barrier(){MPI_Barrier(MPI_COMM_WORLD);}

void mpi_env::Bcast_Tensor(Tensor* t_in, int root){
    size_t sz = t_in->size;
    float* data = new float [sz];
    MPI_Bcast(data, sz, MPI_FLOAT, root, MPI_COMM_WORLD);
}

float mpi_env::Gather_and_average(float send_data){
    float average = 0.0;	
    MPI_Gather(&send_data, 1, MPI_FLOAT, avg_data, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < mpi_size; i++)
       average += *(avg_data + i);	    
    return average * div;
}

void mpi_env::Allreduce_Tensor(Tensor* t_in){
    size_t sz = t_in->size;
    size_t block = mpi_block;
    size_t mits = sz/block + 1;
    size_t rem = sz%block;

    float* out_ptr_h_start = new float [sz];
    float* out_ptr_h = out_ptr_h_start;
    float* in_ptr_h = new float [sz];

    // blocked all_reduce + rescale
    for (size_t mit=0; mit<mits; ++mit){
        // if last block go through reminder
        if (mit==mits-1)
            block = rem;
        float* out_beg = out_ptr_h; // save beginning of block
        MPI_Allreduce(in_ptr_h, out_ptr_h, block, MPI_FLOAT,
              MPI_SUM, MPI_COMM_WORLD);
        out_ptr_h = out_beg; // rewind block of output
        //for(size_t i=0; i<block; ++i)
        //    *(out_ptr_h++) *= div; // rescale
    }

    delete [] out_ptr_h_start;
    delete [] in_ptr_h;
}

void mpi_env::Broadcast_params(Net* net){
    vlayer layers = net->layers;
    for (unsigned int i = 0; i < layers.size(); i++) {
        if (layers[i]->trainable) {
            for (int j = 0; j < layers[i]->get_trainable_params_count(); j++) {
                Tensor* par = layers[i]->params[j];
                Bcast_Tensor(par, 0);
            }
        }
    }
}
