#!/bin/bash
cd /home/sgd_mpi/data
mkdir -p imagenette2-224 && cd imagenette2-224
wget 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
tar xvfz imagenette2-320.tgz
rm imagenette2-320.tgz
mv imagenette2-320 imagenette2-224

## Crop and resize to 224x224
cd imagenette2-224
find . -name '*.JPEG' | xargs -L100 -P4 mogrify -gravity center -crop 320x320+0+0 -resize 224x224
cd ..

## yaml file creation
python3 /home/sgd_mpi/code/utils/create_yaml_dataset.py /home/sgd_mpi/data/imagenette2-224/imagenette2-224/train/ \
       	/home/sgd_mpi/data/imagenette2-224/imagenette2-224/val/ \
	--name imagenette --out_filename /home/sgd_mpi/data/imagenette2-224/imagenette2-224.yaml
