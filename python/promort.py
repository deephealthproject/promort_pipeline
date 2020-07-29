"""\
PROMORT example.
"""

import argparse
import random
import sys

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

import models 

def VGG16(in_layer, num_classes):
    x = in_layer
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 64, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 128, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 256, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 512, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 512, [3, 3])), [2, 2], [2, 2])
    x = eddl.Reshape(x, [-1])
    x = eddl.ReLu(eddl.Dense(x, 256))
    x = eddl.Softmax(eddl.Dense(x, num_classes))
    return x


def main(args):
    num_classes = 2
    size = [256, 256]  # size of images

    in_ = eddl.Input([3, size[0], size[1]])
    out = VGG16(in_, num_classes)
    net = eddl.Model([in_], [out])
    eddl.build(
        net,
        eddl.rmsprop(1e-6),
        #eddl.sgd(0.0001, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU([1]) if args.gpu else eddl.CS_CPU()
    )
    eddl.summary(net)
    eddl.setlogfile(net, "promort_VGG16_classification")
    
    training_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size),
        ecvl.AugMirror(.5),
        ecvl.AugFlip(.5),
        ecvl.AugRotate([-180, 180]),
        ecvl.AugAdditivePoissonNoise([0, 10]),
        ecvl.AugGammaContrast([0.5, 1.5]),
        ecvl.AugGaussianBlur([0, 0.8]),
        ecvl.AugCoarseDropout([0, 0.3], [0.02, 0.05], 0.5)
    ])
    
    validation_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size),
    ])
    
    dataset_augs = ecvl.DatasetAugmentations(
        [training_augs, validation_augs, None]
    )

    print("Reading dataset")
    d = ecvl.DLDataset(args.in_ds, args.batch_size)
    #d = ecvl.DLDataset(args.in_ds, args.batch_size, dataset_augs)
    x = Tensor([args.batch_size, d.n_channels_, size[0], size[1]])
    y = Tensor([args.batch_size, len(d.classes_)])
    num_samples_train = len(d.GetSplit())
    num_batches_train = num_samples_train // args.batch_size

    d.SetSplit(ecvl.SplitType.validation)
    num_samples_val = len(d.GetSplit())
    num_batches_val = num_samples_val // args.batch_size
    
    indices = list(range(args.batch_size))
    metric = eddl.getMetric("categorical_accuracy")

    print("Starting training")
    
    ### Main loop across epochs
    for e in range(args.epochs):
        print("Epoch {:d}/{:d} - Training".format(e + 1, args.epochs),
              flush=True)
        if args.out_dir:
            current_path = os.path.join(args.out_dir, "Epoch_%d" % e)
            for c in d.classes_:
                c_dir = os.path.join(current_path, c)
                os.makedirs(c_dir, exist_ok=True)

        d.SetSplit(ecvl.SplitType.training)
        eddl.reset_loss(net)
        total_metric = []
        s = d.GetSplit()
        random.shuffle(s)
        d.split_.training_ = s
        d.ResetAllBatches()
        
        ### Looping across batches of training data
        for b in range(num_batches_train):
            print("Epoch {:d}/{:d} (batch {:d}/{:d}) - ".format(
                e + 1, args.epochs, b + 1, num_batches_train
            ), end="", flush=True)
            d.LoadBatch(x, y)
            x.div_(255.0)
            tx, ty = [x], [y]
            #print (tx[0].info())
            eddl.train_batch(net, tx, ty, indices)
            eddl.print_loss(net, b)
            print()

        print("Saving weights")
        eddl.save(net, "promort_checkpoint_%s.bin" % e, "bin")

        ### Evaluation on validation set
        print("Epoch %d/%d - Evaluation" % (e + 1, args.epochs), flush=True)
        d.SetSplit(ecvl.SplitType.validation)
        for b in range(num_batches_val):
            n = 0
            print("Epoch {:d}/{:d} (batch {:d}/{:d}) - ".format(
                e + 1, args.epochs, b + 1, num_batches_val
            ), end="", flush=True)
            d.LoadBatch(x, y)
            x.div_(255.0)
            eddl.forward(net, [x])
            output = eddl.getTensor(out)
            sum_ = 0.0
            for k in range(args.batch_size):
                result = output.select([str(k)]))
                target = y.select([str(k)])
                ca = metric.value(target, result)
                total_metric.append(ca)
                sum_ += ca
                if args.out_dir:
                    result_a = np.array(result, copy=False)
                    target_a = np.array(target, copy=False)
                    classe = np.argmax(result_a).item()
                    gt_class = np.argmax(target_a).item()
                    single_image = x.select([str(k)])
                    img_t = ecvl.TensorToView(single_image)
                    img_t.colortype_ = ecvl.ColorType.BGR
                    single_image.mult_(255.)
                    filename = d.samples_[d.GetSplit()[n]].location_[0]
                    head, tail = os.path.splitext(os.path.basename(filename))
                    bname = "%s_gt_class_%s.png" % (head, gt_class)
                    cur_path = os.path.join(
                        current_path, d.classes_[classe], bname
                    )
                    ecvl.ImWrite(cur_path, img_t)
                n += 1
            print("categorical_accuracy:", sum_ / args.batch_size)
        total_avg = sum(total_metric) / len(total_metric)
        print("Total categorical accuracy:", total_avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_ds", metavar="INPUT_DATASET")
    parser.add_argument("--epochs", type=int, metavar="INT", default=50)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out-dir", metavar="DIR",
                        help="if set, save images in this directory")
    main(parser.parse_args())
