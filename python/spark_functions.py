from cassandra.auth import PlainTextAuthProvider
from cassandra_dataset import CassandraDataset
import time
from tqdm import trange, tqdm
import numpy as np
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

def set_net_weights(net, new_weights):
    l_l = net.layers
    for i, l in enumerate(l_l):
        w_l = new_weights[i]
        if w_l:
            bias = w_l[0]
            weights = w_l[1]
            l.update_weights(bias, weights)
            eddl.distributeParams(l)


def get_num_batches(el, seed):
    def ret(split_index):
        # Environment loading
        el.value.start(seed)
        cd = el.value.cd
        return(cd.num_batches[split_index])
    return(ret)


def train(el, index_list, init_weights, sync_iterations, lr, gpus,
          dropout, l2_reg, seed):
    def ret(split_index):
        print('Starting train function')
        t0 = time.time()

        # Environment loading
        el.value.start(seed)
        cd = el.value.cd
        cd.set_indexes(index_list)
        net = el.value.net
        out = net.layers[-1]

        t1 = time.time()
        print("Time to load the Environment  %.3f" % (t1-t0))
        t0 = t1

        # Updating model weights
        if isinstance(init_weights.value, str):
            eddl.load(net, str(init_weights.value))
            t1 = time.time()
            print("Time to Load weights %.3f" % (t1-t0))
            t0 = t1

        elif init_weights.value:
            new_weights = init_weights.value
            print(new_weights[-2][0].shape)

            t1 = time.time()
            print("Receiving data from broadcast: %.3f" % (t1-t0))
            t0 = t1

            # updating net weights layer by layer
            new_weights = [[Tensor(i) for i in l] for l in new_weights]
            set_net_weights(net, new_weights)

            t1 = time.time()
            print(
                "Time to convert np array to tensor and set parameter to the net: %.3f" % (t1-t0))
            t0 = t1

        ###################
        ## Training step ##
        ###################
        metric_fn = eddl.getMetric("categorical_accuracy")
        loss_fn = eddl.getLoss("soft_cross_entropy")

        # Main loop across epochs
        # Training
        if index_list[split_index] == 0:  # Epoch start
            eddl.reset_loss(net)

        metric_l = []
        loss_l = []

        # Looping across batches of training data
        pbar = tqdm(range(sync_iterations))

        for b_index, b in enumerate(pbar):
            x, y = cd.load_batch(split_index)

            x.div_(255.0)
            tx, ty = [x], [y]
            eddl.train_batch(net, tx, ty)

            # print bratch train results
            loss = eddl.get_losses(net)[0]
            metr = eddl.get_metrics(net)[0]
            msg = "Batch {:d}/{:d}) - loss: {:.3f}, acc: {:.3f}".format(
                b + 1, sync_iterations, loss, metr)
            pbar.set_postfix_str(msg)
            loss_l.append(loss)
            metric_l.append(metr)

        pbar.close()

        t1 = time.time()
        print("Time to perform training  %.3f" % (t1-t0))
        t0 = t1

        # End of training. Get Loss, metric and weights
        p = eddl.get_parameters(net)  # Get parameters from the model
        # Transform tensors to numpy array for spark serialization
        r = [[i.getdata() for i in l] for l in p]

        t1 = time.time()
        print("Time to get parameters and convert them to numpy array  %.3f" % (t1-t0))
        t0 = t1

        res = (r, sum(loss_l), sum(metric_l))
        return res
    return ret


def sum_weights(in0, in1):
    # Weight lists, losses list, metric list
    w0, l0, m0 = in0
    w1, l1, m1 = in1

    print("Reduce Function")
    t0 = time.time()
    tot_weights = [[(w0[i][j]+w1[i][j]) for j, _ in enumerate(l)]
                   for i, l in enumerate(w0)]

    tot_losses = l0 + l1

    tot_metrics = m0 + m1

    t1 = time.time()
    print("Reduce function time: %.3f" % (t1-t0))

    return (tot_weights, tot_losses, tot_metrics)

