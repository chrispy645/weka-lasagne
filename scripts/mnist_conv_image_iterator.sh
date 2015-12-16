#!/bin/bash

java -Xmx6g weka.Run .LasagneNet \
    -S 1 \
    -layer ".Conv2DLayer -filter_size_x 5 -filter_size_y 5 -num_filters 10 -nonlinearity .Rectify" \
    -layer ".MaxPool2DLayer -pool_size_x 2 -pool_size_y 2" \
    -layer ".Conv2DLayer -filter_size_x 3 -filter_size_y 3 -num_filters 10 -nonlinearity .Rectify" \
    -layer ".MaxPool2DLayer -pool_size_x 2 -pool_size_y 2" \
    -layer ".DenseLayer -num_units 10 -nonlinearity .Rectify" \
    -loss ".CategoricalCrossEntropy -l1 0.0 -l2 0.0" \
    -update ".Adagrad -learning_rate 0.1 -epsilon 1.0E-6" \
    -epochs 10 \
    -batch_iterator ".ImageBatchIterator -batch_size 128 -shuffle -width 28 -height 28 -prefix '/Users/cjb60/github/weka-lasagne/datasets/mnist-data'" \
    -eval_size 0.1 \
    -out_file /tmp/out.log \
    -t ../datasets/mnist.meta.arff \
    -no-cv \
    -output-debug-info
