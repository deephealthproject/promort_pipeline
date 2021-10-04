

## Running the example:
```bash
cd /home/sgd_mpi/code/examples/imagenette2-224/python
mpirun --bind-to none -n 2 --hostfile /home/sgd_mpi/code/hostfile python3 mpi_training.py \ 
       --yml-in /home/sgd_mpi/data/imagenette2-224/imagenette2-224.yaml --gpu 1 1 --batch-size 28
```
