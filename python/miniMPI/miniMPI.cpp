#include <iostream>
#include "miniMPI.hpp"

miniMPI::miniMPI(int bl){
  int h_len;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Get_processor_name(hostname, &h_len);
  mpi_hostname = hostname;
  cout << "Hello from " << mpi_hostname << ", rank " <<
    mpi_rank << " of " << mpi_size << endl;
  div = 1/(static_cast<float>(mpi_size));
  mpi_block = bl*1024;
}

miniMPI::~miniMPI(){
  MPI_Finalize();
}

void miniMPI::LoLAverage(LoL& input, LoL& output){
  for (size_t lev=0; lev!=input.size(); ++lev){
    auto in = input[lev];
    auto out = output[lev];
    for (size_t idx=0; idx<in.size(); ++idx){
      auto np_in = in[idx];
      auto np_out = out[idx];
      auto buf_in = np_in.request();
      auto buf_out = np_out.request();
      float* ptr_in = static_cast<float *>(buf_in.ptr);
      float* ptr_out = static_cast<float *>(buf_out.ptr);
      size_t sz = np_in.size();
      size_t block = mpi_block;
      size_t mits = sz/block + 1;
      size_t rem = sz%block;
      // blocked all_reduce + rescale
      for (size_t mit=0; mit<mits; ++mit){
	// if last block go through reminder
	if (mit==mits-1)
	  block = rem;
	float* out_beg = ptr_out; // save beginning of block
	MPI_Allreduce(ptr_in, ptr_out, block, MPI_FLOAT,
		      MPI_SUM, MPI_COMM_WORLD);
	ptr_out = out_beg; // rewind block of output
	for(size_t i=0; i<block; ++i)
	  *(ptr_out++) *= div; // rescale
      }
    }
  }
}

void miniMPI::Barrier(){
  MPI_Barrier(MPI_COMM_WORLD);
}

void miniMPI::LoLBcast(LoL& data, int root){
  for (auto lev=data.begin(); lev!=data.end(); ++lev){
    for (auto np=lev->begin(); np!=lev->end(); ++np){
      auto buf = np->request();
      float* d_ptr = static_cast<float *>(buf.ptr);
      size_t sz = np->size();
      MPI_Bcast(d_ptr, sz, MPI_FLOAT, root, MPI_COMM_WORLD);
    }
  }
}

void miniMPI::Gather(float x, cpyar& ret, int root){
  auto buf = ret.request();
  auto d_ptr = buf.ptr;
  MPI_Gather(&x, 1, MPI_FLOAT, d_ptr, mpi_size, MPI_FLOAT,
	     root, MPI_COMM_WORLD);
}
