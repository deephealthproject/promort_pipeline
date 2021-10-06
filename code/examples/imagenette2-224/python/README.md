## Running the example:
```bash
export HALF_CORES=$(python3 /home/sgd_mpi/code/utils/half_cores.py)
```

without data augmentations (1 node, 2 GPUs):
```bash
cd /home/sgd_mpi/code/examples/imagenette2-224/python
mpirun --mca pml ucx --map-by node:pe=1 --bind-to core -n 2 --hostfile /home/sgd_mpi/code/hostfile python3 mpi_training.py --yml-in /home/sgd_mpi/data/imagenette2-224/imagenette2-224.yaml --gpu 1 1 --batch-size 28
```

with data augmentations (1 node, 2 GPUs):
```bash
cd /home/sgd_mpi/code/examples/imagenette2-224/python
mpirun --mca pml ucx --map-by node:pe=$HALF_CORES --bind-to core -n 2 --hostfile /home/sgd_mpi/code/hostfile python3 mpi_training.py --yml-in /home/sgd_mpi/data/imagenette2-224/imagenette2-224.yaml --gpu 1 1 --batch-size 28 --augs-on
```

without data augmentations (1 node, 1 GPU):
```bash
cd /home/sgd_mpi/code/examples/imagenette2-224/python
mpirun --mca pml ucx --map-by node:pe=1 --bind-to core -n 1 --hostfile /home/sgd_mpi/code/hostfile python3 mpi_training.py --yml-in /home/sgd_mpi/data/imagenette2-224/imagenette2-224.yaml --gpu 1 --batch-size 28 
```

in general the program options are:
```
usage: mpi_training.py [-h] --yml-in DIR [--epochs INT]
                       [--sync-iterations INT] [--patch-size INT] [--augs-on]
                       [--patience INT] [--batch-size INT] [--seed INT]
                       [--lr FLOAT] [--dropout FLOAT] [--l2-reg FLOAT]
                       [--gpu GPU [GPU ...]] [--save-weights] [--out-dir DIR]
                       [--net-name NET_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --yml-in DIR          yaml input file
  --epochs INT          Number of total epochs
  --sync-iterations INT
                        Number of step between weights sync
  --patch-size INT      Patch side size
  --augs-on             Activate data augmentation
  --patience INT        Number of epochs after which the training is stopped
                        if validation accuracy does not improve (delta=0.001)
  --batch-size INT      Batch size
  --seed INT            Seed of the random generator to manage data load
  --lr FLOAT            Learning rate
  --dropout FLOAT       Float value (0-1) to specify the dropout ratio
  --l2-reg FLOAT        L2 regularization parameter
  --gpu GPU [GPU ...]   Specify GPU mask. For example: 1 to use only gpu0; 1 1
                        to use gpus 0 and 1; 1 1 1 1 to use gpus 0,1,2,3
  --save-weights        Network parameters are saved after each epoch
  --out-dir DIR         Specifies the output directory. If not set no output
                        data is saved
  --net-name NET_NAME   Select a model between resnet50 and vgg16
  ```
