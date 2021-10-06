/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"
#include "optim_sgd_mpi.hpp"
#include "mpi_env.hpp"
//#include <mpi.h>

using namespace eddl;

//////////////////////////////////
// mnist_mlp.cpp:
// A very basic MLP for mnist
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv) {
    int n_sync = 1;
    mpi_env* MPE = new mpi_env(n_sync);
    int mpi_size = MPE->mpi_size;
    int mpi_rank = MPE->mpi_rank;

    bool testing = false;
    bool use_cpu = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }

    // Download mnist
    download_mnist();

    // Settings
    int epochs = (testing) ? 2 : 10;
    int batch_size = 200;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = LeakyReLu(Dense(l, 1024));
    l = LeakyReLu(Dense(l, 1024));
    l = LeakyReLu(Dense(l, 1024));

    layer out = Softmax(Dense(l, num_classes), -1);  // Softmax axis optional (default=-1)
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        std::vector<int> gpu_mask(mpi_size, 0);
	gpu_mask[mpi_rank % mpi_size] = 1;
	cs = CS_GPU(gpu_mask, "full_mem"); // one GPU
    }

    SGD_mpi* opt = new SGD_mpi(MPE, 0.01, 0.9, 0.0, false);

    // Build model
    build(net,
          opt, // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          cs );

    // Initializing all network copies with the same parameters
    MPE->Broadcast_params(net);

    // View model
    summary(net);
    
    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    Tensor* y_train = Tensor::load("mnist_trY.bin");
    Tensor* x_test = Tensor::load("mnist_tsX.bin");
    Tensor* y_test = Tensor::load("mnist_tsY.bin");

    // Computing the size of dataset to be processed by the current rank
    int train_data_size = x_train->getShape()[0];
    //int test_data_size = x_test->getShape()[0];

    int per_rank_train_ds = train_data_size / mpi_size;
    int start = per_rank_train_ds * mpi_rank;
    std::string _range_ = std::to_string(start) + ":" + std::to_string(start + per_rank_train_ds);
    
    Tensor* x_rank_train = x_train->select({_range_, ":"});
    Tensor* y_rank_train = y_train->select({_range_, ":"});
    
    delete x_train;
    delete y_train;

    // Preprocessing
    x_rank_train->div_(255.0f);
    x_test->div_(255.0f);

    // Train model
    fit(net, {x_rank_train}, {y_rank_train}, batch_size, epochs);

    // Evaluate: Only rank 0 perform evaluation on the test set
    if (mpi_rank == 0)
    	evaluate(net, {x_test}, {y_test});

    // Release objects, layers, optimizer and computing service are released by the net object
    delete x_rank_train;
    delete y_rank_train;
    delete x_test;
    delete y_test;
    delete net;
    
    delete MPE;
    //MPI_Finalize(); 
    return EXIT_SUCCESS;
}
