# weka-lasagne
WekaLasagne is a wrapper for the neural network library Lasagne, which allows users to easily construct neural networks
using WEKA's GUI.

![gui](https://raw.githubusercontent.com/chrispy645/weka-lasagne/master/images/gui.png)

## Examples

This repository comes with the MNIST digits dataset in ARFF format. We can easily define a multilayer perceptron and train it, using 66% of the data for training and the rest for testing, like so:

```
java weka.Run .LasagneNet \
     -layer ".DenseLayer -num_units 50 -nonlinearity .Rectify" \
     -loss ".CategoricalCrossEntropy" \
     -update ".Momentum -learning_rate 0.01 -momentum 0.9" -epochs 100 -bs 128 \
     -out_file "/tmp/output.txt" \
     -t datasets/mnist.arff -split-percentage 66 
```

