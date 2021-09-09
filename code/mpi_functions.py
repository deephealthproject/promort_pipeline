from cassandra.auth import PlainTextAuthProvider
from cassandra_dataset import CassandraDataset
import time
from tqdm import trange, tqdm
import numpy as np
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import random
import pickle, os

def train(el, init_weights_fn, epochs, lr, gpus, dropout, l2_reg, seed, out_dir):
    
    MP = el.MP
    rank = MP.mpi_rank

    print('Starting train function')
    t0 = time.time()

    # Get Environment
    el.start(seed)
    cd = el.cd
    net = el.net
    out = net.layers[-1]

    t1 = time.time()
    print("Time to load the Environment  %.3f" % (t1-t0))
    t0 = t1

    # Loading model weights if any
    if init_weights_fn:
        eddl.load(net, init_weights_fn)

    ###################
    ## Training step ##
    ###################
    
    print("Starting training", flush=True)

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
    t0 = time.time()
    
    ## Make the split setup before the main loop. Then mix only training splits
    #el.split_setup(seed)
    train_splits_l = [iii for iii in range(0, el.num)]
    val_splits_l = [iii for iii in range(el.num, 2*el.num)]
    tr_num_batches = min(cd.num_batches[0:el.num]) - 1 # FIXME: Using the minimum among all batches not the local ones
    val_num_batches = min(cd.num_batches[el.num:el.num*2]) - 1 # FIXME: Using the minimum among all batches not the local ones
    
    print ("NUM BATCHES %r" % cd.num_batches)
    
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
        cd.mix_splits(train_splits_l)
    
        pbar = tqdm(range(tr_num_batches))
        
        for b_index, mb in enumerate(pbar):
            # Init local weights to a zero structure equal in size and shape to the global one
            t0 = time.time()
            split_index = rank # Local split. If ltr == 1 --> split_index = rank
                
            x, y = cd.load_batch(split_index)

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
                msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) - loss: {:.3f}, acc: {:.3f}".format(e + 1, epochs, tr_num_batches + 1, b_index, loss, acc)
                pbar.set_postfix_str(msg)
                epoch_loss_l.append(loss)
                epoch_acc_l.append(acc)
            
        ## End of macro batches
        pbar.close()
        
        ######
        ## Validation 
        ######
        if rank == 0:
            print ("\n\n\n*** Evaluation ***")

        eddl.reset_loss(net)
        seed = random.getrandbits(32)
        for sp in val_splits_l:
            cd.rewind_splits(sp)
        
        epoch_val_acc_l = []
        epoch_val_loss_l = []
        
        pbar = tqdm(range(val_num_batches))
        
        for b_index, mb in enumerate(pbar):
            # Init local weights to a zero structure equal in size and shape to the global one
            t0 = time.time()
            split_index = el.num + rank # Local validation split
                
            x, y = cd.load_batch(split_index)
            x.div_(255.0)
            tx, ty = [x], [y]
                    
            #print (f'Train batch rank: {rank}, ep: {e}, macro_batch: {mb}, local training rank: {lt}, inidipendent iteration: {s_it}') 
            eddl.forward(net, tx)
            
            net_out = eddl.getOutput(net.layers[-1]) 
       
            sum_ca = 0.0 ## sum of samples accuracy within a batch
            sum_ce = 0.0 ## sum of losses within a batch

            n = 0
            for k in range(x.getShape()[0]):
                result = net_out.select([str(k)])
                target = y.select([str(k)])
                ca = acc_fn.value(target, result)
                ce = loss_fn.value(target, result)
                sum_ca += ca
                sum_ce += ce
                n += 1
            
            loss = (sum_ce / n)
            acc = (sum_ca / n)
            
            # Loss and accuracy synchronization among ranks
            loss = MP.Gather_and_average(loss)
            acc = MP.Gather_and_average(acc)

            if rank == 0:
                # Only rank 0 print progression bar
                epoch_val_loss_l.append(loss)
                epoch_val_acc_l.append(acc)
                msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) - loss: {:.3f}, acc: {:.3f}".format(e + 1, epochs, val_num_batches + 1, b_index, loss, acc)
                pbar.set_postfix_str(msg)
            
        ## End of macro batches
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
