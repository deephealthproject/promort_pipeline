"""
PROMORT example.
"""

import argparse
import random
import sys
import os
import io
import PIL.Image

sys.path.append('../')

from pyeddl.tensor import Tensor

from cassandra_dataset import CassandraDataset

from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import trange, tqdm
import numpy as np
import pickle 


def main(args):
    num_classes = 2
    
    #################################
    ### Set database to read data ###
    #################################

    if not args.cassandra_pwd_fn:
        cass_pass = getpass('Insert Cassandra password: ')	
    else:
        with open(args.cassandra_pwd_fn) as fd:
            cass_pass = fd.readline().rstrip()

    # create cassandra reader
    ap = PlainTextAuthProvider(username='prom', password=cass_pass)
    cd = CassandraDataset(ap, ['127.0.0.1'])

    #cd.init_listmanager(meta_table='promort.osk_ids', id_col='patch_id',
    #                    split_ncols=2, num_classes=num_classes, 
    #                    partition_cols=['sample_name', 'sample_rep', 'label'])
    
    cd.load_splits(args.splits_fn, batch_size=32, augs=[])
        
    n_splits = len(cd.split); # 2: 0 train, 1 val; #3: 0 train, 1 val, 2 test;
    split_names_d = {0:'train', 1:'val', 2:'test'}
    class_d = {1:'normal', 2:'tumor'}

    ### Create output folders
    for s_name in split_names_d.values():
        for c_name in class_d.values():
            os.makedirs(os.path.join(args.out_dir, s_name, c_name), exist_ok=True)

    
    for si in split_names_d.keys():
        split_name = split_names_d[si]
        num_batches = cd.num_batches[si]
        cd.current_split = si ## Set the current split 
    
        #print ("Current split: %d, %s" % (si, split_name), flush=True)
        cd.rewind_splits(shuffle=False)
        rows = cd.row_keys[cd.split[si]]
        n_rows = len(rows)
        
        pbar = tqdm(range(n_rows))
       
        query = f"SELECT {cd.id_col}, {cd.label_col}, {cd.data_col} FROM {cd.table} WHERE {cd.id_col}=?"
        prep = cd._clm.sess.prepare(query)

        pbar = tqdm(rows)
        for r_index, r in enumerate(pbar):
            key = rows[r_index]
            res = cd._clm.sess.execute(prep, key, execution_profile='tuple')
            data = res.one()
            idx = '%s.jpg' % str(data[0])
            lab = data[1]
            raw_img = data[2]
            c_name = class_d[lab]

            in_stream = io.BytesIO(raw_img)
            img = PIL.Image.open(in_stream)
           
            target_fname = os.path.join(args.out_dir, split_name, c_name, idx)
            img.save(target_fname)

            msg = "batch {:d}/{:d})".format(r_index + 1, n_rows)
            pbar.set_postfix_str(msg)
        
        pbar.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-fn", metavar="STR", required=True,
                        help="split filename. It is pickle file")
    parser.add_argument("--out-dir", metavar="STR", 
                        help="output directory where images are written", default="out_cassandra_data")
    parser.add_argument("--cassandra-pwd-fn", metavar="STR",
                        help="cassandra password")
    main(parser.parse_args())
