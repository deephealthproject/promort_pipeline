#include "optim_sgd_mpi.hpp"

SGD_mpi::SGD_mpi(mpi_env* MPE, float lr, float momentum, float weight_decay, bool nesterov):
	SGD(lr, momentum, weight_decay, nesterov), MPE(MPE) {
    n_sync = MPE->n_sync;
    count = 0;
    lr /= (static_cast<float>(MPE->mpi_size)); // Normalization for the ALLReduce Operation 
    
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
            Tensor::add(lr, layers[i]->gradients[j], mu, mT[p], mT[p], 0);
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
		MPE->Allreduce_Tensor(t_in);
            }
        }
    }
}


// High level API
optimizer sgd_mpi(mpi_env* MPE, float lr, float momentum, float weight_decay, bool nesterov) {
    return new SGD_mpi(MPE, lr, momentum, weight_decay, nesterov);
}
