#!/bin/bash

java weka.Run .LasagneNet \
     -layer ".DenseLayer -num_units 2 -nonlinearity .Rectify" \
     -loss ".CategoricalCrossEntropy" \
     -update ".Momentum -learning_rate 0.01 -momentum 0.9" -epochs 10000 -bs 100000 \
     -t ../datasets/iris.arff -no-cv
