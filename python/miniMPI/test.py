# run with
# mpirun -n 2 --hostfile hostfile python3 test.py

import numpy as np
import time
import pickle 
from MMPI import miniMPI
import sys
import time

def main():
    MP = miniMPI()
    with open("./vgg16_np_weights.pckl", "rb") as fin:
        x = pickle.load(fin)
    y = []
    for i in x:
        a = []
        for j in i:
            a += [np.empty_like(j)]
        y += [a]
    MP.Barrier()
    for _ in range(100):
        t0 = time.time()
        MP.LoLAverage(x, y)
        t1 = time.time()
        if MP.mpi_rank == 0:
            print(f'{t1-t0:.3f}')
            #print(f'{x[-2][1]} --> {y[-2][1]}')

if __name__ == "__main__":
    main()
