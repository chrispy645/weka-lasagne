from __future__ import print_function

import lasagne
from lasagne import *
from lasagne.updates import *
from lasagne.nonlinearities import *
from lasagne.layers import *
from lasagne.objectives import *

from nolearn.lasagne import *
from pyscript.pyscript import *

import gzip
import os
import numpy as np
import sys
import re

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

def remove_colour(st):
    ansi_escape = re.compile(r'\x1b[^m]*m')
    return ansi_escape.sub('', st)

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout

def get_net(args):
    ##GET_NET##

def train(args):
    if args["regression"]:
        y_train = np.asarray(args["y_train"], dtype="float32")
    else:
        y_train = np.asarray(args["y_train"].flatten(), dtype="int32")
    net1 = get_net(args)
    X_train = np.asarray(args["X_train"], dtype="float32")
    X_train = X_train.reshape( (X_train.shape[0], 1, X_train.shape[1]) )
    with Capturing() as output:
        model = net1.fit(X_train, y_train)
    return { "results": remove_colour("\n".join(output)),
        "params": net1.get_all_params_values() }

def describe(args, model):
    return model["results"]

def test(args, model):
    net1 = get_net(args)
    net1.load_params_from(model["params"])
    X_test = np.asarray(args["X_test"], dtype="float32")
    X_test = X_test.reshape( (X_test.shape[0], 1, X_test.shape[1]) )
    return net1.predict_proba(X_test).tolist()

if __name__ == "__main__":
    f = ArffToArgs()
    f.set_input("")
    args = f.get_args()
    f.close()
    dd = train(args)
    print(dd["results"])
