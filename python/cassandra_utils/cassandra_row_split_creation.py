"""
PROMORT example.
"""

import argparse
import sys
import os
from pathlib import Path

from cassandradl import CassandraDataset

from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
import numpy as np
import pickle 

def main(args):
    print ("Requested Splits from:")
    print ("data_table: %s" % args.data_table)
    print ("ids_table: %s" % args.ids_table)
    print ("metatable: %s" % args.metatable)
    print ("splits: %r" % args.split_ratios)
    print ("grouping_cols: %r" % args.grouping_cols)



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
    cd = CassandraDataset(ap, ['cassandra-db'])

    grouping_cols = ['sample_name']

    if args.lab_map:
        lab_map = [int(i) for i in args.lab_map]
    else:
        lab_map = []

    if args.balance:
        if len(args.balance) != args.num_classes:
            print ("\nError: The number of balance elements must be equal to num_classes")
            sys.exit(-1)
        balance = [float(i) for i in args.balance]
    else:
        balance = None
    
    cd.init_listmanager(table='promort.' + args.ids_table, 
                        metatable='promort.' + args.metatable, 
                        id_col='patch_id',
                        num_classes=args.num_classes, 
                        label_col=args.label,
                        label_map = lab_map,
                        grouping_cols=args.grouping_cols)

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

    data_size = args.data_size
    
    #if args.balanced:
    #    min_class = np.min(clm._stats.sum(axis=0))
    #    data_size = min(data_size, min_class*clm.num_classes)
    #    print (clm._stats.sum(axis=0))
    #    print (min_class)
   
    print (clm._stats.sum(axis=0), clm.tot)
    
    ### Check for bags
    if args.bags_pckl:
        bags = pickle.load(open(args.bags_pckl, 'rb'))
        print (bags)
    else:
        bags = None

    ## Create the split and save it
    cd.init_datatable(table='promort.' + args.data_table)
    cd.split_setup(batch_size=args.batch_size, split_ratios=args.split_ratios,
                                 max_patches=data_size, augs=[], bags=bags, balance=balance)

    print (clm._stats.sum(axis=0), clm.tot)
    print (clm._split_stats)
    print (clm._split_stats_actual)

    
    out_fn = os.path.join(args.out_dir, '%s.pckl' % args.out_name)
    print ("Saving splits in: %s" % out_fn)
    cd.save_splits(out_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-table", default="data_cosk_0")
    parser.add_argument("--ids-table", default="ids_cosk_0")
    parser.add_argument("--metatable", default="metatable_cosk_0")
    parser.add_argument("--bags-pckl", default=None)
    parser.add_argument("--grouping_cols", nargs='+', default=['sample_name'])
    parser.add_argument("--label", default='label')
    parser.add_argument("--num-classes", type=int, metavar="INT", default=2)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32)
    parser.add_argument("--split-ratios", nargs='+', type=int, default="7 2 1")
    parser.add_argument("--data-size", type=int, metavar="INT", default=None, help='Specifies the total number of patches to be used to create splits. If not specified all patches are used')
    parser.add_argument("--lab-map", nargs='+', default = [], help='Specifies label mapping. It is used to group original dataset lebel to new class labels. For example: if the original dataset has [0, 1, 2, 3] classes using lab-map 0 0 1 1 maps the new label 0 to the old 0,1 classes and the new label 1 to the old 2,3 classes ')
    parser.add_argument("--balance", nargs='+', default = [], help='Specifies balance among classes. The number of values must be equal to the number of classes. For example: if the dataset has three classes, using --balance 1 1 1 set the same number of patches for all classes.')
    parser.add_argument("--out-dir", metavar="DIR", default='/tmp',
                        help="destination folder")
    parser.add_argument("--out-name", metavar="DIR", default = "splits.pckl",
                        help="name of the output pickle file with requested splits")
    parser.add_argument("--db-rows-fn", metavar="STR",
                        help="load db rows from a pickle file if it exists or save rows to a pickle after reading image metadata from db")
    parser.add_argument("--cassandra-pwd-fn", metavar="STR", default='/tmp/cassandra_pass.txt',
                        help="cassandra password")
    main(parser.parse_args())
