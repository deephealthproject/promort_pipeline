#include "data_generator/data_generator.h"
#include "models/models.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

using namespace ecvl;
using namespace eddl;
using namespace std;
using namespace std::filesystem;

int main(int argc, char* argv[])
{
    // Settings
    int epochs = 50;
    int batch_size = 32;
    int num_classes = 2;
    std::vector<int> size{ 256,256 }; // Size of images

    vector<int> gpus = { 1 };
    int lsb = 1;
    string mem = "low_mem";
    string checkpoint = "";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--low-mem")) {
            mem = "low_mem";
        }
        else if (!strcmp(argv[i], "--mid-mem")) {
            mem = "mid_mem";
        }
        else if (!strcmp(argv[i], "--full-mem")) {
            mem = "full_mem";
        }
        else if (!strcmp(argv[i], "--lsb")) {
            lsb = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--batch-size")) {
            batch_size = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--gpus-2")) {
            gpus = { 1,1 };
        }
        else if (!strcmp(argv[i], "--gpus-1")) {
            gpus = { 1 };
        }
        else if (!strcmp(argv[i], "--checkpoint")) {
            checkpoint = argv[++i];
        }
    }

    std::mt19937 g(std::random_device{}());

    // Define network
    layer in = Input({ 3, size[0],  size[1] });
    layer out = VGG16_promort(in, num_classes);
    model net = Model({ in }, { out });

    // Build model
    build(net,
	rmsprop(0.000001f),
        //sgd(0.001f, 0.9f), // Optimizer
        { "soft_cross_entropy" }, // Losses
        { "categorical_accuracy" }, // Metrics
        CS_GPU(gpus, lsb, mem) // Computing Service
    );

    if (!checkpoint.empty()) {
        load(net, checkpoint, "bin");
    }

    // View model
    summary(net);
    plot(net, "model.pdf");
    setlogfile(net, "skin_lesion_classification");

    DatasetAugmentations dataset_augmentations{ {nullptr, nullptr, nullptr } };

    // Read the dataset
    cout << "Reading dataset" << endl;
    DLDataset d("/DeepHealth/git/promort_pipeline/python/set0_small_med.yaml", batch_size, move(dataset_augmentations));
    
    // Create producer thread with 'DLDataset d' and 'std::queue q'
    int num_samples = vsize(d.GetSplit());
    int num_batches = num_samples / batch_size;
    DataGenerator d_generator_t(&d, batch_size, size, { vsize(d.classes_) }, 3);

    d.SetSplit(SplitType::validation);
    int num_samples_validation = vsize(d.GetSplit());
    int num_batches_validation = num_samples_validation / batch_size;
    DataGenerator d_generator_v(&d, batch_size, size, { vsize(d.classes_) }, 2);

    float sum = 0., ca = 0.;

    tensor output;

    vector<int> indices(batch_size);
    iota(indices.begin(), indices.end(), 0);
    cv::TickMeter tm;
    cv::TickMeter tm_epoch;

    cout << "Starting training" << endl;
    for (int i = 0; i < epochs; ++i) {

        tm_epoch.reset();
        tm_epoch.start();

        d.SetSplit(SplitType::training);
        // Reset errors
        reset_loss(net);

        // Shuffle training list
        shuffle(std::begin(d.GetSplit()), std::end(d.GetSplit()), g);
        d.ResetAllBatches();

        d_generator_t.Start();

        // Feed batches to the model
        for (int j = 0; d_generator_t.HasNext() /* j < num_batches */; ++j) {
            tm.reset();
            tm.start();
            cout << "Epoch " << i << "/" << epochs << " (batch " << j << "/" << num_batches << ") - ";
            cout << "|fifo| " << d_generator_t.Size() << " - ";

            tensor x, y;

            // Load a batch
            if (d_generator_t.PopBatch(x, y)) {
                // Preprocessing
                x->div_(255.0);
		
                // Train batch
                //train_batch(net, { x }, { y }, indices);
		forward(net, { x });
		output = getOutput(out);
		cout << output->select({ to_string(0) }) << endl;

                // Print errors
                print_loss(net, j);

                delete x;
                delete y;
            }
            tm.stop();


            cout << "Elapsed time: " << tm.getTimeSec() << endl;
        }

        d_generator_t.Stop();
        tm_epoch.stop();
        cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;
	
    }
	
    delete output;

    return EXIT_SUCCESS;
}
