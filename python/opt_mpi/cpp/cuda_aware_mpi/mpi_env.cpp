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

    std::cout << "MPI_ENV Constructor, hello from " << hostname << ", rank " << mpi_rank << std::endl;
}

mpi_env::~mpi_env(){MPI_Finalize();}

void mpi_env::Barrier(){MPI_Barrier(MPI_COMM_WORLD);}

void mpi_env::Bcast_Tensor(Tensor* t_in, int root){
    std::cout << "BROADCAST, rank " << mpi_rank << ", " << t_in->ptr << std::endl;
    MPI_Bcast(t_in->ptr, t_in->size, MPI_FLOAT, root, MPI_COMM_WORLD);
}

void mpi_env::Allreduce_Tensor(Tensor* t_in){
    float* t_in_ptr = t_in->ptr;
    size_t sz = t_in->size;
    float* t_out_ptr = new float [sz];
    size_t block = mpi_block;
    size_t mits = sz/block + 1;
    size_t rem = sz%block;

    std::cout << "SYNC_GRAD: " << mpi_rank << ", "  << t_in->ptr << ", " << t_out_ptr << ", " << sz << ", " << block << ", " << mits << ", " << rem << std::endl;
    // blocked all_reduce + rescale
    for (size_t mit=0; mit<mits; ++mit){
        // if last block go through reminder
        if (mit==mits-1)
            block = rem;
        float* out_beg = t_out_ptr; // save beginning of block
        std::cout << "LOOP " << mit << ", " << block << std::endl;
        MPI_Allreduce(t_in_ptr, t_out_ptr, block, MPI_FLOAT,
              MPI_SUM, MPI_COMM_WORLD);
        std::cout << "AllReduce done" << mit << std::endl;
        t_out_ptr = out_beg; // rewind block of output
        for(size_t i=0; i<block; ++i)
            *(t_out_ptr++) *= div; // rescale
    }
    delete t_out_ptr;
    std::cout << "EXIT" << std::endl;
}
