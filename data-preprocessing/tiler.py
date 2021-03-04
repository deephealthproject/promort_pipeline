from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.conf import SparkConf
from pyspark import StorageLevel
import os
import io
from getpass import getpass
import openslide
from PIL import Image
import numpy as np
import uuid
from tissue_detector import tissue_detector as td

# Run with
# /spark/bin/spark-submit --conf spark.cores.max=48 --conf spark.default.parallelism=48 --py-files tissue_detector.py,LSVM_tissue_bg_model_promort.pickle tiler.py
#
# For testing in ipython:
# set -x PYSPARK_DRIVER_PYTHON ipython3
# /spark/bin/pyspark --conf spark.cores.max=48 --conf spark.default.parallelism=48 --master spark://spark-master:7077
#
# If cassandra gets flooded and needs some oxygen:
# cassandra.yaml: write_request_timeout_in_ms: 30000

# pip3 install cassandra-driver
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
from cassandra.cluster import ExecutionProfile

slide_root = '/data/o/slides'
masks_root = '/data/o/masks'
ext = '.mrxs'
pyram_lev = 2

class CassandraWriter():
    def __init__(self, auth_prov, cassandra_ips, table1, table2, table3):
        prof = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory = cassandra.query.dict_factory)
        profs = {'default': prof}
        self.cluster = Cluster(cassandra_ips,
                               execution_profiles=profs,
                               protocol_version=4,
                               auth_provider=auth_prov)
        self.sess = self.cluster.connect()
        query1 = f"INSERT INTO {table1} (sample_name, sample_rep, x, y, label, tissue, patch_id) VALUES (?,?,?,?,?,?,?)"
        query2 = f"INSERT INTO {table2} (patch_id, label, data) VALUES (?,?,?)"
        query3 = f"INSERT INTO {table3} (sample_name, sample_rep, x, y, label, tissue, patch_id) VALUES (?,?,?,?,?,?,?)"
        self.prep1 = self.sess.prepare(query1)
        self.prep2 = self.sess.prepare(query2)
        self.prep3 = self.sess.prepare(query3)
    def __del__(self):
        self.cluster.shutdown()
    def save_items(self, items):
        # buffer of futures
        for item in items:
            # if buffer full pop two elements from top
            sample_name, sample_rep, x, y, patch_id, label, tissue, data = item
            # if not enough tissue skip
            if (tissue<.3):
                continue;
            i1 = self.sess.execute_async(self.prep1, (sample_name, sample_rep,
                                                      x, y, label, tissue, patch_id),
                                         execution_profile='default',
                                         timeout=30)
            i3 = self.sess.execute_async(self.prep3, (sample_name, sample_rep,
                                                      x, y, label, tissue, patch_id),
                                         execution_profile='default',
                                         timeout=30)
            # wait for remaining async inserts to finish
            i1.result(); i3.result()
            # insert heavy data synchronously
            self.sess.execute(self.prep2, (patch_id, label, data),
                              execution_profile='default',
                              timeout=30)

class Tiler():
    def __init__(self, sample, slide_fn, mask_norm_fn=None,
                 mask_tum_fn=None, pyram_lev=0):
        self.slide_fn = slide_fn
        self.sample_name, srep = sample.split('-')
        self.sample_rep = int(srep)
        self.mask_norm_fn = mask_norm_fn
        self.mask_tum_fn = mask_tum_fn
        self.pyram_lev = pyram_lev # setting level from which patches are read
        self.patch_x, self.patch_y = (256, 256)
        self.slide = openslide.OpenSlide(self.slide_fn)
        self.ts_cl = td(model_fn='LSVM_tissue_bg_model_promort.pickle',
                       th=0.8)
    def __del__(self):
        self.slide.close()
    def get_tile(self, x, y, label):
        patch = self.slide.read_region((x,y), self.pyram_lev,
                                     (self.patch_x, self.patch_y))
        patch = patch.convert('RGB')
        # compute tissue coverage
        ar = np.asarray(patch)
        t_mask = self.ts_cl.get_tissue_mask(ar)
        tissue = t_mask.mean()
        # generate jpeg
        out_stream = io.BytesIO()
        patch.save(out_stream, format='JPEG', quality=90)
        # write to db
        out_stream.flush()
        data = out_stream.getvalue()
        patch_id = uuid.uuid4()
        return (self.sample_name, self.sample_rep, x, y, patch_id, label, tissue, data)
    def get_tiles(self, coords):
        for (x,y), label in coords:
            yield(self.get_tile(x, y, label))
    def get_coords(self):
        norm = Image.open(self.mask_norm_fn)
        tum = Image.open(self.mask_tum_fn)
        d0 = np.array(self.slide.dimensions)
        dL = np.array(self.slide.level_dimensions[self.pyram_lev])
        up_x, up_y = d0/dL
        ox, oy = d0
        nx, ny = round(ox/(self.patch_x*up_x)), round(oy/(self.patch_y*up_y))
        sc_x = ox/nx; sc_y = oy/ny
        mini_norm = norm.resize((nx, ny))
        mini_tum = tum.resize((nx, ny))
        # label choice
        lab_norm = 1 # 0b01
        lab_tum = 2  # 0b10
        coords_norm = [((round(sc_x*x), round(sc_y*y)), lab_norm) for x in
                       range(nx) for y in range(ny)
                       if mini_norm.getpixel((x,y))==1]
        coords_tum = [((round(sc_x*x), round(sc_y*y)), lab_tum) for x in
                      range(nx) for y in range(ny)
                      if mini_tum.getpixel((x,y))==1]
        norm.close(); tum.close()
        return (coords_norm + coords_tum)

def get_job_list(sample):
    max_job_size = 1000
    slide_fn = os.path.join(slide_root, sample+ext)
    mask_norm_fn = os.path.join(masks_root, 'normal', sample+'_mask.png')
    mask_tum_fn = os.path.join(masks_root, 'tumor', sample+'_mask.png')
    tiler = Tiler(sample, slide_fn, mask_norm_fn, mask_tum_fn,
                  pyram_lev=pyram_lev)
    coords = tiler.get_coords()
    def chunks(lst, m):
        n=len(lst)
        if (n==0):
            return(lst)
        c=(n+m-1)//m
        sl=(n+c-1)//c
        for i in range(0, len(lst), sl):
            yield lst[i:i + sl]
    return(sample, list(chunks(coords, max_job_size)))

def get_tiles(params):
    sample, coords = params
    slide_fn = os.path.join(slide_root, sample+ext)
    mask_norm_fn = os.path.join(masks_root, 'normal', sample+'_mask.png')
    mask_tum_fn = os.path.join(masks_root, 'tumor', sample+'_mask.png')
    tiler = Tiler(sample, slide_fn, mask_norm_fn, mask_tum_fn,
                  pyram_lev=pyram_lev)
    return (tiler.get_tiles(coords))

def write_to_cassandra(password):
    def ret(items):
        auth_prov = PlainTextAuthProvider('prom', password)
        cw = CassandraWriter(auth_prov, ['cassandra_db'],
                             f'promort.ids_cosk_{pyram_lev}',
                             f'promort.data_cosk_{pyram_lev}',
                             f'promort.metadata_cosk_{pyram_lev}',)
        cw.save_items(items)
    return(ret)
    
def run():
    try: 
        from private_data import cass_pass
    except ImportError:
        cass_pass = getpass('Insert Cassandra password: ')

    conf = SparkConf()\
        .setAppName("Tiler")\
        .setMaster("spark://spark-master:7077")
    #conf.set('spark.scheduler.mode', 'FAIR')
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    
    parts_0 = 48 # get list of patches
    parts_1 = 24 # extract patches
    parts_2 = 24 # write to cassandra

    samples = next(os.walk(os.path.join(masks_root,'normal')))[2]
    samples = [s.split('_')[0] for s in samples]
    par_samples = sc.parallelize(samples, numSlices=parts_0)
    cols = ['sample_name', 'sample_rep', 'x', 'y', 'label', 'data', 'patch_id']
    data = par_samples\
        .map(get_job_list)\
        .flatMapValues(lambda x: x)\
        .repartition(parts_1)\
        .flatMap(get_tiles)
    # save to Cassandra tables
    data.coalesce(parts_2)\
        .foreachPartition(write_to_cassandra(cass_pass))
    

# uncomment to use with spark-submit
run()
