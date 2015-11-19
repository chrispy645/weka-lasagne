import lasagne
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.layers import *
from lasagne.updates import *
import theano
from theano import tensor as T
import numpy as np

def prepare(args):
    X = T.fmatrix('X')
    y = T.ivector('y')
    output_shapes = []
    ##NETWORK##
    all_params = lasagne.layers.get_all_params(out_layer)
    prediction = lasagne.layers.get_output(out_layer, X)
    ##LOSS##
    return {
        "X": X,
        "y": y,
        "out_layer": out_layer,
        "all_params": all_params,
        "loss": loss,
        "prediction": prediction,
        "output_shapes": output_shapes,
    }

def train(args):
    
    symbols = prepare(args)
    X = symbols["X"]
    y = symbols["y"]
    all_params = symbols["all_params"]
    prediction = symbols["prediction"]
    loss = symbols["loss"]
    out_layer = symbols["out_layer"]
    output_shapes = symbols["output_shapes"]
    
    X_train = np.asarray(args["X_train"], dtype="float32")
    y_train = np.asarray(args["y_train"].flatten(), dtype="int32")
    
    ##UPDATES##
    label_vector = prediction
    
    iter_train = theano.function(
        [X, y],
        [label_vector],
        updates=updates
    )

    bs = args["batch_size"]
    num_epochs = args["num_epochs"]
    batch_train_losses = []
    for e in range(0, num_epochs):
        b = 0
        while True:
            if b*bs >= X_train.shape[0]:
                break
            X_train_batch = X_train[b*bs : (b+1)*bs]
            y_train_batch = y_train[b*bs : (b+1)*bs]
            this_loss = iter_train(X_train_batch, y_train_batch)
            batch_train_losses.append(this_loss)
            b += 1
    
    return ( output_shapes, get_all_param_values(out_layer) )

def describe(args, model):
    desc = []
    for st in model[0]:
        desc.append( str(st) )
    ##DESCRIBE##
    return "\n".join(desc)

def test(args, model):
    symbols = prepare(args)
    out_layer = symbols["out_layer"]
    X = symbols["X"]
    label_vector = symbols["prediction"]
    
    lasagne.layers.set_all_param_values(out_layer, model[1])

    X_test = np.asarray(args["X_test"], dtype="float32")

    iter_test = theano.function(
        [X],
        label_vector
    )

    preds = iter_test(X_test).tolist()

    #print preds
    
    return preds
