#!/bin/bash

java weka.Run .LasagneNet \
     -layer ".DenseLayer -num_units 2 -nonlinearity .Rectify" \
     -loss ".CategoricalCrossEntropy" \
     -update ".Adagrad -learning_rate 0.1" \
     -batch_iterator ".BatchIterator -batch_size 10000" \
     -epochs 10000 -S 0 \
     -t ../datasets/iris.arff -no-cv
