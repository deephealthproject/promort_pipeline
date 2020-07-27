"""\
PROMORT example.
"""

import argparse
import random
import sys
import os
import glob

import yaml

def main(args):
    train_path = args.train_dir
    val_path = args.val_dir
    test_path = args.test_dir
    name = args.name
    desc = args.desc
    out_fn = args.out_filename

    if not name:
        name = train_path
    if not desc:
        desc = train_path

    train_classes = sorted(os.listdir(train_path))
    val_classes = sorted(os.listdir(val_path))
    if test_path:
        test_classes = sorted(os.listdir(test_path))
    else:
        test_classes = val_classes

    if not (train_classes == val_classes and test_classes == val_classes):
        raise("Training and validation classes must be the same")

    # YAML substructures creation
        # Images
    images_l = []
    
    n_train = 0
    for c in train_classes:
        fnames = glob.glob(os.path.join(train_path, c, "*"))
        images_l +=[{'location': l, 'label':c} for l in fnames]
        n_train += len(fnames)

    n_val = 0
    for c in val_classes:
        fnames = glob.glob(os.path.join(val_path, c, "*"))
        images_l += [{'location': l, 'label':c} for l in fnames]
        n_val += len(fnames)

    if test_path:
        n_test = 0
        for c in test_classes:
            fnames = glob.glob(os.path.join(test_path, c, "*"))
            images_l += [{'location': l, 'label':c} for l in fnames]
            n_test = len(fnames)

        # Split
    train_indexes = [i for i in range(n_train)]
    val_indexes = [i+n_train for i in range(n_val)]
    
    if test_path:
        test_indexes = [i+n_train+n_val for i in range(n_test)]
        split_dict = {'training':train_indexes, 'validation':val_indexes, 'test': test_indexes}
    else:
        split_dict = {'training':train_indexes, 'validation':val_indexes}
    

    # YAML dictionary
    y_dict = {'name': name, 'description': desc, 'classes': train_classes, 'images': images_l, 'split': split_dict}
    
    yaml.dump(y_dict, open(out_fn, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_dir", type=str, metavar="train_path")
    parser.add_argument("val_dir", type=str, metavar="val_path")
    parser.add_argument("--out_filename", type=str, metavar="out_filename", default='out.yaml')
    parser.add_argument("--test_dir", type=str, metavar="test_path [optional]", default=None)
    parser.add_argument("--name", type=str, metavar="dataset name [optional]", default=None)
    parser.add_argument("--desc", type=str, metavar="dataset description [optional]", default=None)
    main(parser.parse_args(sys.argv[1:]))
