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
import pickle 

def main(args):
    print ("Requested Splits from:")
    print ("table: %s" % args.table)
    print ("meta-table: %s" % args.metatable)
    print ("splits: %r" % args.splits)



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
    #cd = CassandraDataset(ap, ['cassandra_db'])
    cd = CassandraDataset(ap, ['127.0.0.1'])

    cd.init_listmanager(meta_table='promort.' + args.metatable, id_col='patch_id',
                        split_ncols=args.split_ncols, num_classes=args.num_classes, 
                        partition_cols=args.partition_cols)
    
    data_size = args.data_size
    
    if args.db_rows_fn:
        if Path(args.db_rows_fn).exists():
            # If rows file exists load them from file 
            cd.load_rows(args.db_rows_fn)
        else:
            # If rows do not exist, read them from db and save to a pickle
            cd.read_rows_from_db()
            cd.save_rows(args.db_rows_fn)
    else:
        # If a db_rows_fn is not specified just read them from db
        cd.read_rows_from_db()
        
    ## Create the split and save it
    cd.init_datatable(table='promort.' + args.table)
    cd.split_setup(batch_size=args.batch_size, split_ratios=args.splits,
                                 max_patches=data_size, augs=[])

    out_fn = os.path.join(args.out_dir, args.out_name)
    print ("Saving splits in: %s" % out_fn)
    cd.save_splits(out_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="data_osk")
    parser.add_argument("--metatable", default="ids_osk")
    parser.add_argument("--partition_cols", nargs='+', default="['sample_name', 'sample_rep', 'label']")
    parser.add_argument("--split-ncols", type=int, metavar="INT", default=1)
    parser.add_argument("--num-classes", type=int, metavar="INT", default=2)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32)
    parser.add_argument("--splits", nargs='+', type=int, default="[7, 2, 1]")
    parser.add_argument("--data-size", type=int, metavar="INT", default=100000)
    parser.add_argument("--out-dir", metavar="DIR", default='/tmp',
                        help="destination folder")
    parser.add_argument("--out-name", metavar="DIR", default = "splits.pckl",
                        help="name of the output pickle file with requested splits")
    parser.add_argument("--db-rows-fn", metavar="STR",
                        help="load db rows from a pickle file if it exists or save rows to a pickle after reading image metadata from db")
    parser.add_argument("--cassandra-pwd-fn", metavar="STR",
                        help="cassandra password")
    main(parser.parse_args())
