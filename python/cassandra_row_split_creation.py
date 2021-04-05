"""
PROMORT example.
"""

import argparse
import sys
import os
from pathlib import Path

from cassandra_dataset import CassandraDataset

from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
import numpy as np
import pickle 

def main(args):
    print ("Requested Splits from:")
    print ("table: %s" % args.table)
    print ("ids_table: %s" % args.ids_table)
    print ("metatable: %s" % args.metatable)
    print ("splits: %r" % args.split_ratios)
    print ("partition_cols: %r" % args.partition_cols)



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

    cd.init_listmanager(metatable='promort.' + args.metatable,
                        table = 'promort.' + args.ids_table,
                        id_col='patch_id',
                        split_ncols=args.split_ncols, 
                        num_classes=args.num_classes, 
                        partition_cols=args.partition_cols)
    
    if args.db_rows_fn:
        if Path(args.db_rows_fn).exists():
            # If rows file exists load them from file 
            cd.load_rows(args.db_rows_fn)
        else:
            # If rows do not exist, read them from db and save to a pickle
            cd.read_rows_from_db()
            cd.save_rows(os.path.join(args.out_dir, '%s.pckl' % args.db_rows_fn))
    else:
        # If a db_rows_fn is not specified just read them from db
        cd.read_rows_from_db()
    
    clm = cd._clm

    data_size = min(cd._clm.tot, args.data_size)
    if args.balanced:
        min_class = np.min(clm._stats.sum(axis=0))
        data_size = min(data_size, min_class*clm.num_classes)
    
    print ("data size: %d" % data_size)
    
    ### Check for bags
    if args.bags_pckl:
        bags = pickle.load(open(args.bags_pckl, 'rb'))
        print (bags)
    else:
        bags = None

    ## Create the split and save it
    cd.init_datatable(table='promort.' + args.table)
    cd.split_setup(batch_size=args.batch_size, split_ratios=args.split_ratios,
                                 max_patches=data_size, augs=[], bags=bags)

    out_fn = os.path.join(args.out_dir, '%s.pckl' % args.out_name)
    print ("Saving splits in: %s" % out_fn)
    cd.save_splits(out_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="data_cosk_0")
    parser.add_argument("--ids-table", default="ids_cosk_0")
    parser.add_argument("--metatable", default="metatable_cosk_0")
    parser.add_argument("--bags-pckl", default=None)
    parser.add_argument("--partition_cols", nargs='+', default=['sample_name', 'sample_rep', 'label'])
    parser.add_argument("--split-ncols", type=int, metavar="INT", default=1)
    parser.add_argument("--num-classes", type=int, metavar="INT", default=2)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32)
    parser.add_argument("--split-ratios", nargs='+', type=int, default="[7, 2, 1]")
    parser.add_argument("--data-size", type=int, metavar="INT", default=100000)
    parser.add_argument("--balanced", action="store_true", help='returns balanced splits')
    parser.add_argument("--out-dir", metavar="DIR", default='/tmp',
                        help="destination folder")
    parser.add_argument("--out-name", metavar="DIR", default = "splits.pckl",
                        help="name of the output pickle file with requested splits")
    parser.add_argument("--db-rows-fn", metavar="STR",
                        help="load db rows from a pickle file if it exists or save rows to a pickle after reading image metadata from db")
    parser.add_argument("--cassandra-pwd-fn", metavar="STR", default='/tmp/cassandra_pass.txt',
                        help="cassandra password")
    main(parser.parse_args())
