import argparse
from cassandra.auth import PlainTextAuthProvider
from cassandra_dataset import CassandraDataset
from getpass import getpass
from pyspark import StorageLevel
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import time
from tqdm import trange, tqdm
import io
import numpy as np
import os

# Run with
# /spark/bin/spark-submit --py-files cassandra_dataset.py,BPH.cpython-36m-x86_64-linux-gnu.so,/home/cesco/code/tmp/inet_256_rows.pckl --conf spark.cores.max=10 dist_train.py --nodes 10

def train(inet_pass, num, init_val, seed=123):
    def ret(i):
        ap = PlainTextAuthProvider(username='inet', password=inet_pass)
        cd = CassandraDataset(ap, ['cassandra_db'], port=9042, seed=seed)
        cd.load_rows('inet_256_rows.pckl')
        cd.init_datatable(table='imagenet.data_256')
        cd.split_setup(batch_size=32, split_ratios=[1]*num,
                       max_patches=1300000, augs=[])
        x,y = cd.load_batch(i)
        r = x.getdata()[0,0,0,0]
        r *= init_val.value
        return r
    return ret
    
def run(args):
    try: 
        from private_data import inet_pass
    except ImportError:
        inet_pass = getpass('Insert Cassandra password: ')

    conf = SparkConf()\
        .setAppName("Distributed training")\
        .setMaster("spark://spark-master:7077")
    #conf.set('spark.scheduler.mode', 'FAIR')
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    
    num = args.nodes # number of nodes
    w = 1
    
    nodes = range(num)
    par_nodes = sc.parallelize(nodes, numSlices=num)
    for ep in range(5):
        init_val = sc.broadcast(w)
        data = par_nodes\
            .map(train(inet_pass, num, init_val))\
            .reduce(lambda x,y: x+y)
        avg = data/num
        print(avg)
        w = avg/100

if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, metavar="INT", default=3,
                        help='Number of nodes')
    run(parser.parse_args())
