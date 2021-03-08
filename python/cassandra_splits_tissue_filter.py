import argparse
import sys, os

from cassandra_dataset import CassandraDataset
from cassandra.auth import PlainTextAuthProvider
from getpass import getpass

from tqdm import tqdm
import numpy as np
import pickle 

def main(args):
    cassandra_pwd_fn = args.cassandra_pwd_fn
    tissue_th_min = args.tissue_th_min
    tissue_th_max = args.tissue_th_max
    splits_fn = args.splits_fn
    
    ## Create output path
    if args.out_fn:
        out_splits_fn = args.out_fn
    else:
        bn, ext = os.path.splitext(os.path.basename(splits_fn))
        path = os.path.dirname(splits_fn)
        tmp = bn + "_tcr_%.2f_%.2f" % (tissue_th_min, tissue_th_max) + ext
        out_splits_fn = os.path.join(path, tmp)
    
    print ("Writing to: %s" % out_splits_fn)

    ## Open Cassandra session and load splits
    with open(cassandra_pwd_fn) as fd:
        cass_pass = fd.readline().rstrip()
    ap = PlainTextAuthProvider(username='prom', password=cass_pass)
    cd = CassandraDataset(ap, ['127.0.0.1'])

    cd.load_splits(splits_fn, batch_size=32, augs=[])
    
    ## Filter data
    query = f"SELECT * FROM {cd.metatable} WHERE {'patch_id'}=?"
    prep = cd._clm.sess.prepare(query)

    new_row_keys_l = [[] for i in range(cd.num_splits)]

    for si in range(cd.num_splits):
        cd.current_split = si ## Set the current split 
        rows = cd.row_keys[cd.split[si]]

        pbar = tqdm(rows[:])
        for r_index, r in enumerate(pbar):
            key = rows[r_index]
            res = cd._clm.sess.execute(prep, key, execution_profile='tuple')
            data = res.one()
            # patch_id, class, sample_name, sample_rep, tissue coverage ratio, x, y

            idx = data[0]
            tcr = data[4]

            if (tcr >= tissue_th_min) and (tcr <= tissue_th_max):
                new_row_keys_l[si].append({'patch_id': idx})

        pbar.close()
        
    ## New cassandra fields related to filtered splits 
    new_row_keys_l_flat = [item for sublist in new_row_keys_l for item in sublist]
    new_row_keys = np.array(new_row_keys_l_flat)

    ## New n
    n_l = np.array([len(i) for i in new_row_keys_l])
    new_n = np.sum(n_l)

    ## New splits
    start = 0
    new_split = []
    for index, i in enumerate(n_l):
        new_split.append(np.array(range(start, i+start)))
        start = new_split[index][-1] + 1
    
    ## Update split data
    cd.n = new_n
    cd.split = new_split
    cd.row_keys = new_row_keys

    ## Save new split
    cd.save_splits(out_splits_fn)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-fn", metavar="STR", required=True,
                        help="split filename. It is pickle file")
    parser.add_argument("--out-fn", metavar="STR", 
                        help="output directory where images are written", default="")
    parser.add_argument("--cassandra-pwd-fn", metavar="STR",
                        help="cassandra password", default="/tmp/cassandra_pass.txt")
    parser.add_argument("--tissue-th-min", type=float, help="min tissue coverage ratio requested", default=0.8)
    parser.add_argument("--tissue-th-max", type=float, help="max tissue coverage ratio requested", default=1.0)
    
    main(parser.parse_args())
