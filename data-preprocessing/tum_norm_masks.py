from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import mir
import os

def mask_kernel(args) :
    slide, xmlroot, mask_dir, basename, lab, suf, xmlfile = args
    if (not os.path.exists(slide)) :
        print(f"{slide} does not exist")
        return
    xmldir = list(os.walk(xmlroot))[1][0]
    xml = os.path.join(xmldir, xmlfile)
    reader = mir.MRimage()
    reader.open(slide)
    reader.XML2TIFF_3(xml, mask_dir, basename, lab, suf)

def mask_gen(lab, suf, xmlfile, srcdir, slidedir, maskdir) :
    dlist = os.scandir(srcdir)
    basenames = [e.name for e in dlist if e.is_dir()]
    # run parallel jobs
    pool = mp.Pool(48)
    args = map(lambda basename : (os.path.join(slidedir, basename + '.mrxs'),
                                  os.path.join(srcdir, basename),
                                  maskdir, basename, lab, suf, xmlfile),
               basenames)
    pool.map(mask_kernel, args)
    pool.close()

def gen_all(srcdir = '/data/promort/rois.test',
            slidedir='/data/promort/prom2/slides',
            maskdir='/data/promort/masks.test') :


    # generate masks for normal cores
    mask_gen('_2', 'mask', 'cores.xml', srcdir, slidedir, maskdir)
    # generate masks for tumoral roi's
    mask_gen('_0', 'mask', 'focus_regions.xml', srcdir, slidedir, maskdir)

## Interactive pyspark
# set -x PYSPARK_DRIVER_PYTHON ipython3
# /spark/bin/pyspark --conf spark.cores.max=24 --conf spark.default.parallelism=24 --py-files mir.py --master spark://spark-master:7077
#
## Run program
# /spark/bin/spark-submit --conf spark.cores.max=24 --conf spark.default.parallelism=24 --py-files mir.py run.py

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
    work = [['_2', 'cores.xml'], ['_0', 'focus_regions.xml']]
    suf = 'mask'
    
    dlist = os.scandir(srcdir)
    basenames = [e.name for e in dlist if e.is_dir()]

    # build job list
    job_list = []
    for basename in basenames:
        for lab, xmlfile in work:
            slide = os.path.join(slidedir, basename + '.mrxs')
            xmlroot = os.path.join(srcdir, basename)
            job_list.append((slide, xmlroot, maskdir, basename, lab,
                             suf, xmlfile))
    procs = sc.defaultParallelism
    rdd = sc.parallelize(job_list, numSlices=procs)
    rdd.foreach(mask_kernel)
    
    
# run main
spark_run()
