#include "optim_sgd_mpi.hpp"

SGD_mpi::SGD_mpi(mpi_env* MPE, float lr, float momentum, float weight_decay, bool nesterov):
	SGD(lr, momentum, weight_decay, nesterov), MPE(MPE) {
    n_sync = MPE->n_sync;
    count = 0;
     
    // Broadcast parameters of rank 0 to the other ones
    sync_rank_0_parameters();
    // Barrier to sync all workers
    MPE->Barrier();
}

SGD_mpi::~SGD_mpi(){
}

Optimizer* SGD_mpi::clone(){
    SGD_mpi *n = new SGD_mpi(MPE, lr, mu, weight_decay, nesterov);
    n->clip_val=clip_val;

    return n;
}

//Optimizer* SGD_mpi::share() override{
//
//}

void SGD_mpi::applygrads(int batch){
    if (isshared) {
      orig->applygrads(batch);
    }
    else
    {
      if (!(count % n_sync)) {
          // Sync among workers
	  sync_grads();
      	  count = 0; 
      }
      clip();
      int p = 0;
      for (unsigned int i = 0; i < layers.size(); i++) {
        if (layers[i]->trainable) {
          for (int j = 0; j < layers[i]->get_trainable_params_count(); j++, p++) {
            Tensor::add(lr , layers[i]->gradients[j], mu, mT[p], mT[p], 0);
            Tensor::add(1.0, layers[i]->params[j], -1.0, mT[p], layers[i]->params[j], 0);

            // Distributed training: Accumulation of gradients
            if (layers[i]->acc_gradients.size() > 0) 
              Tensor::add(1.0, layers[i]->acc_gradients[j], -1.0, mT[p], layers[i]->acc_gradients[j], 0);
          }
        }
        else p+=layers[i]->get_trainable_params_count();
      }
    }
    count++;
}

void SGD_mpi::sync_grads(){
    for (unsigned int i = 0; i < layers.size(); i++) {
        if (layers[i]->trainable) {
            for (int j = 0; j < layers[i]->get_trainable_params_count(); j++) {
                Tensor* t_in = layers[i]->gradients[j]; // Gradient 
		//Tensor* t_out = new Tensor(t_in->getShape(), layers[i]->dev);
		//MPE->Allreduce_Tensor(t_in);
		Allreduce_Tensor(t_in);
		// Copy to layer gradient
                //Tensor::copy(t_out, t_in);
            }
        }
    }
}

void SGD_mpi::sync_rank_0_parameters(){
    for (unsigned int i = 0; i < layers.size(); i++) {
        if (layers[i]->trainable) {
            for (int j = 0; j < layers[i]->get_trainable_params_count(); j++) {
                Tensor* par = layers[i]->params[j];
                MPE->Bcast_Tensor(par, 0);
            }
        }
    }
}

void SGD_mpi::Allreduce_Tensor(Tensor* t_in){
    float* t_in_ptr = t_in->ptr;
    size_t sz = t_in->size;
    Tensor* t_out = new Tensor(t_in->getShape(), 1000);
    float* t_out_ptr = t_out->ptr;
    size_t block = MPE->mpi_block;
    size_t mits = sz/block + 1;
    size_t rem = sz%block;

    std::cout << "SYNC_GRAD: " << t_in->ptr << ", " << t_out_ptr << ", " << sz << ", " << block << ", " << mits << ", " << rem << std::endl;
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
            *(t_out_ptr++) *= 2.0; // rescale
    }
    Tensor::copy(t_out, t_in);
    delete t_out;
    std::cout << "EXIT" << std::endl;
}

// High level API
optimizer sgd_mpi(mpi_env* MPE, float lr, float momentum, float weight_decay, bool nesterov) {
    return new SGD_mpi(MPE, lr, momentum, weight_decay, nesterov);
}
