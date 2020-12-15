from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from PIL import Image, ImageDraw
import numpy as np
import openslide
import json
import os
import csv


## Interactive pyspark
# set -x PYSPARK_DRIVER_PYTHON ipython3
# /spark/bin/pyspark --conf spark.cores.max=24 --conf spark.default.parallelism=24 --master spark://spark-master:7077
#
## Run program
# /spark/bin/spark-submit --conf spark.cores.max=24 --conf spark.default.parallelism=24 run.py

# scan focus_regions or cores csv
def scan_csv(csvdir, csvfile, filter_lab) :
    csvpath = os.path.join(csvdir, csvfile)
    if (not os.path.exists(csvpath)):
        return []
    with open(csvpath) as infile:
        dr = csv.DictReader(infile)
        if (csvfile=='focus_regions.csv'):
            for row in dr :
                fn = row['file_name']
                ts = row['tissue_status']
                if (ts==filter_lab):
                    yield(fn)
        elif (csvfile=='cores.csv'):
            for row in dr :
                fn = row['file_name']
                ts = 'TUMOR' if (int(row['focus_regions_count'])>0) else 'NORMAL'
                if (ts==filter_lab):
                    yield(fn)
        else:
            raise ValueError('Only focus_regions or cores can be extracted.')

def mask_kernel(args, scale=64) :
    slide, csvroot, maskdir, basename, lab, suf, csvfile = args
    print(slide, maskdir, basename, lab, suf)
    if (not os.path.exists(slide)) :
        print(f"{slide} does not exist")
        return
    csvdir = list(os.walk(csvroot))[1][0]
    wsi = openslide.OpenSlide(slide)
    sx, sy = wsi.dimensions
    wsi.close()
    nx, ny = round(sx/scale), round(sy/scale)
    img = Image.new('L', (nx, ny), 0)
    for fn in scan_csv(csvdir, csvfile, lab):
        csvfn = os.path.join(csvdir,fn)
        with open(csvfn) as jsonfile:
            pol = np.array(json.load(jsonfile))
        pol = pol/scale
        tpol = list(map(tuple, pol))
        ImageDraw.Draw(img).polygon(tpol, fill=1)
    outname = os.path.join(maskdir, lab.lower(), basename + f'_{suf}.png')
    img.save(outname)

def spark_run():
    conf = SparkConf()\
        .setAppName("Mask generator")\
        .setMaster("spark://spark-master:7077")
    #conf.set('spark.scheduler.mode', 'FAIR')
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    srcdir = '/data/promort/rois.test'
    slidedir='/data/promort/prom2/slides'
    maskdir='/data/promort/masks.test' # must have normal and tumor subdirs
    work = [['NORMAL', 'cores.csv'], ['TUMOR', 'focus_regions.csv']]
    suf = 'mask'
    
    dlist = os.scandir(srcdir)
    basenames = [e.name for e in dlist if e.is_dir()]

    # build job list
    job_list = []
    for basename in basenames:
        for lab, csvfile in work:
            slide = os.path.join(slidedir, basename + '.mrxs')
            csvroot = os.path.join(srcdir, basename)
            job_list.append((slide, csvroot, maskdir, basename, lab,
                             suf, csvfile))
    procs = sc.defaultParallelism
    rdd = sc.parallelize(job_list, numSlices=procs)
    rdd.foreach(mask_kernel)
    
    
# run main
spark_run()
