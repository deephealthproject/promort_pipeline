import os, sys

from tqdm import trange, tqdm
import numpy as np
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import random
import pickle
import time

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


def accuracy(predictions, targets, epsilon=1e-12):
    """
    Computes accuracy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    N = predictions.shape[0]
    ce = np.sum((targets * predictions) + 1e-9) / N
    return ce

def train(el, epochs, lr, gpus, dropout, l2_reg, seed, out_dir):
    
    MP = el.MP
    mpi_size = MP.mpi_size
    rank = MP.mpi_rank

    print(f'Starting train function of Rank {rank}')

    # Get Environment
    el.start(seed)
    net = el.net
    yml_fn = el.per_rank_yml_l[rank]
    dataset_augs = el.dataset_augs
    batch_size = el.batch_size 
    size = el.size
    out = net.layers[-1]

    ###################
    ## Training step ##
    ###################
    
    if rank == 0: # Only task 0 takes account of whole stats
        loss_l = []
        acc_l = []
        val_loss_l = []
        val_acc_l = []
        
        patience_cnt = 0
        val_acc_max = 0.0

    acc_fn = eddl.getMetric("categorical_accuracy")
    loss_fn = eddl.getLoss("soft_cross_entropy")
    
    ### Main loop across epochs
    
    ## Make the split setup before the main loop. Then mix only training splits
    #el.split_setup(seed)
    print(f"Rank: {rank} is reading dataset {yml_fn}")

    d = ecvl.DLDataset(yml_fn, batch_size, dataset_augs, num_workers=20)
    #x = Tensor([batch_size, d.n_channels_, size[0], size[1]])
    #y = Tensor([batch_size, len(d.classes_)])
    d.SetSplit(ecvl.SplitType.training)
    num_samples_train = len(d.GetSplit())
    num_batches_train = num_samples_train // batch_size 
    d.SetSplit(ecvl.SplitType.validation)
    num_samples_val = len(d.GetSplit())
    num_batches_val = num_samples_val // batch_size
    
    print (f'Rank {rank}. Train samples: {num_samples_train}, train batches: {num_batches_train}, val samples: {num_samples_val}, val batches: {num_batches_val}')

    for e in range(epochs):
        ####
        ### Training 
        ####
        if rank == 0:
            print ("\n\n\n*** Training ***")
            epoch_acc_l = []
            epoch_loss_l = []
            epoch_val_acc_l = []
            epoch_val_loss_l = []

        ### Recreate splits to shuffle among workers but with the same seed to get same splits
        eddl.reset_loss(net)
        seed = random.getrandbits(32)
        
        ## set split
        d.SetSplit(ecvl.SplitType.training)
        d.ResetBatch(d.current_split_, True);
        d.Start();

        pbar = tqdm(range(num_batches_train * mpi_size))
        
        for mb in range(num_batches_train):
            # Init local weights to a zero structure equal in size and shape to the global one

            t0 = time.time()
            samples, x, y = d.GetBatch()
            
            #d.LoadBatch(x, y)
            t1 = time.time()

            #print (f'load time: {t1-t0}')

            x.div_(255.0)
            tx, ty = [x], [y]
                    
            #print (f'Train batch rank: {rank}, ep: {e}, macro_batch: {mb}, local training rank: {lt}, inidipendent iteration: {s_it}') 
            eddl.train_batch(net, tx, ty)
            
            net_out = eddl.getOutput(net.layers[-1]).getdata() 
            loss = eddl.get_losses(net)[0]
            acc = eddl.get_metrics(net)[0]
        
            # Loss and accuracy synchronization among ranks
            loss = MP.Gather_and_average(loss)
            acc = MP.Gather_and_average(acc)
            
            if rank == 0:
                msg = "Epoch {:d}/{:d} - loss: {:.3f}, acc: {:.3f}".format(e + 1, epochs, loss, acc)
                pbar.set_postfix_str(msg)
                epoch_loss_l.append(loss)
                epoch_acc_l.append(acc)
            
            pbar.update(mpi_size)

        ## End of macro batches
        d.Stop()
        pbar.close()
        
        ######
        ## Validation 
        ######
        if rank == 0:
            print ("\n\n\n*** Evaluation ***")

        eddl.reset_loss(net)
        seed = random.getrandbits(32)
        
        epoch_val_acc_l = []
        epoch_val_loss_l = []
        
        d.SetSplit(ecvl.SplitType.validation) 
        d.ResetBatch(d.current_split_);
        d.Start()
        pbar = tqdm(range(num_batches_val  * mpi_size))
         
        for mb in range(num_batches_val):
            samples, x, y = d.GetBatch()
            x.div_(255.0)
            tx, ty = [x], [y]
                    
            #print (f'Train batch rank: {rank}, ep: {e}, macro_batch: {mb}, local training rank: {lt}, inidipendent iteration: {s_it}') 
            eddl.forward(net, tx)
            
            net_out = eddl.getOutput(net.layers[-1]) 
            
            result = net_out.getdata()
            target = y.getdata()
            sum_ca = accuracy(result, target)
            sum_ce = cross_entropy(result, target)

            loss = (sum_ce / mpi_size)
            acc = (sum_ca / mpi_size)
            
            # Loss and accuracy synchronization among ranks
            loss = MP.Gather_and_average(loss)
            acc = MP.Gather_and_average(acc)
            
            if rank == 0:
                # Only rank 0 print progression bar
                epoch_val_loss_l.append(loss)
                epoch_val_acc_l.append(acc)
                msg = "Epoch {:d}/{:d} - loss: {:.3f}, acc: {:.3f}".format(e + 1, epochs, np.mean(epoch_val_loss_l), np.mean(epoch_val_acc_l))

                pbar.set_postfix_str(msg)
            
            pbar.update(mpi_size)

        ## End of macro batches
        d.Stop()
        pbar.close()
        
        # Compute Epoch loss and acc and store history
        if rank == 0:
            loss_l.append(np.mean(epoch_loss_l))
            acc_l.append(np.mean(epoch_acc_l))
            val_loss_l.append(np.mean(epoch_val_loss_l))
            val_acc_l.append(np.mean(epoch_val_acc_l))

            if out_dir:
                history = {'loss': loss_l, 'acc': acc_l, 'val_loss': val_loss_l, 'val_acc': val_acc_l}
                pickle.dump(history, open(os.path.join(out_dir, 'history.pickle'), 'wb'))
        

    ## End of Epochs
    if rank == 0:
        return loss_l, acc_l, val_loss_l, val_acc_l
    else:
        return None, None, None, None
