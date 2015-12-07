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
    -batch_iterator ".ReshapeBatchIterator -batch_size 128 -width 28 -height 28" \
    -eval_size 0.1 \
    -out_file /tmp/out.log \
    -t ../datasets/mnist.arff \
    -no-cv
