#!/bin/bash

java -Dweka.packageManager.offline=true weka.Run .LasagneNet \
     -layer ".DenseLayer -num_units 50 -nonlinearity .Rectify" \
     -loss ".CategoricalCrossEntropy" \
     -update ".Momentum -learning_rate 0.01 -momentum 0.9" -epochs 100 -bs 128 \
     -out_file "/tmp/output.txt" \
     -t ../datasets/mnist.arff -split-percentage 66 
