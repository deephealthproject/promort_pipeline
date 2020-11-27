from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.sql.functions import broadcast
from PIL import Image, ImageDraw
import csv
import json
import numpy as np
import openslide
import os
import pickle


## Interactive pyspark
# set -x PYSPARK_DRIVER_PYTHON ipython3
# /spark/bin/pyspark --conf spark.cores.max=24 --conf spark.default.parallelism=24 --master spark://spark-master:7077
#
## Run program
# /spark/bin/spark-submit --conf spark.cores.max=24 --conf spark.default.parallelism=24 run.py


def tissue_kernel(b_clf, scale=64) :
    def ret(args) :
        slide, tissuedir, basename, suf = args
        print(slide, tissuedir, basename, suf)
        if (not os.path.exists(slide)) :
            print(f"{slide} does not exist")
            return
        # open whole slide
        wsi = openslide.OpenSlide(slide)
        sx, sy = wsi.dimensions
        # computer target dimensions
        nx, ny = round(sx/scale), round(sy/scale)
        ds = wsi.get_best_level_for_downsample(scale)
        lx, ly = wsi.level_dimensions[ds]
        img = wsi.read_region((0,0), ds, (lx,ly))
        wsi.close()
        img = img.convert('RGB').resize((nx,ny))
        # convert to np array, apply classifier and convert back to PIL.Image
        ar = np.asarray(img)
        n_px = ar.shape[0]*ar.shape[1]
        lin_ar = ar.reshape(n_px, 3)
        clf = b_clf.value
        pred = clf.predict(lin_ar)
        t_mask = pred.reshape(ar.shape[:2])
        outname = os.path.join(tissuedir, basename + f'_{suf}.png')
        t_img = Image.fromarray(t_mask).convert('L')
        t_img.save(outname)
    return(ret)

def spark_run():
    conf = SparkConf()\
        .setAppName("Tissue detector")\
        .setMaster("spark://spark-master:7077")
    #conf.set('spark.scheduler.mode', 'FAIR')
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # broadcast tissue classifier
    with open('code/tmp/LDA_tissue.pickle', 'rb') as pf:
        clf = pickle.load(pf)
    b_clf = sc.broadcast(clf)
    
    srcdir = '/data/promort/rois.test'
    slidedir='/data/promort/prom2/slides'
    tissuedir='/data/promort/tissue.test' # must have normal and tumor subdirs
    suf = 'tissue'
    
    dlist = os.scandir(srcdir)
    basenames = [e.name for e in dlist if e.is_dir()]

    # build job list
    job_list = []
    for basename in basenames:
        slide = os.path.join(slidedir, basename + '.mrxs')
        job_list.append((slide, tissuedir, basename, suf))
    procs = sc.defaultParallelism
    rdd = sc.parallelize(job_list, numSlices=procs)
    # run tissue detector for each slide
    rdd.foreach(tissue_kernel(b_clf))
    
    
# run main
spark_run()
