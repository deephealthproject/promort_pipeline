import argparse
from cassandra.auth import PlainTextAuthProvider
from cassandra_dataset import CassandraDataset
from envloader import EnvLoader
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
# /spark/bin/spark-submit --py-files cassandra_dataset.py,envloader.py,BPH.cpython-36m-x86_64-linux-gnu.so,/home/cesco/code/tmp/inet_256_rows.pckl,/home/cesco/code/tmp/inet_256_10_splits.pckl dist_train.py


def train(cl, num, init_val, indexes, seed=123):
    def ret(i):
        cl.value.start()
        cd = cl.value.cd
        cd.set_indexes(indexes)
        x,y = cd.load_batch(i)
        r = x.getdata()[0,0,0,0]
        r *= init_val.value
        print(cd.current_index)
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
    cl = sc.broadcast(EnvLoader(inet_pass, num))
    for ep in range(30):
        init_val = sc.broadcast(w)
        indexes = list(range(0,num*10,10))
        data = par_nodes\
            .map(train(cl, num, init_val, indexes))\
            .reduce(lambda x,y: x+y)
        avg = data/num
        print(avg)
        w = avg/100

if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, metavar="INT", default=2,
                        help='Number of nodes')
    run(parser.parse_args())
