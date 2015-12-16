from __future__ import print_function
#import lasagne
#from lasagne import *
#from lasagne.updates import *
#from lasagne.nonlinearities import *
#from lasagne.layers import *
#from lasagne.objectives import *
import nolearn.lasagne
from nolearn.lasagne import *
from pyscript.pyscript import *
from skimage import io, img_as_float
import gzip
import os
import numpy as np
import sys
import re
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
import theano
from theano import tensor as T
    
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

def load_image(filename):
    img = io.imread(filename)
    img = img_as_float(img)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = np.asarray( [ img[...,0], img[...,1], img[...,2] ] )
    else:
        img = np.asarray( [ img ] )
    return img
        
def shuffle(*arrays):
    # https://github.com/dnouri/nolearn/issues/27#issuecomment-71175381
    p = np.random.permutation(len(arrays[0]))
    return [array[p] for array in arrays]

class BatchIterator(nolearn.lasagne.BatchIterator):
    # https://github.com/dnouri/nolearn/issues/27#issuecomment-71175381
    def __init__(self, shuffle, *args, **kwds):
        super(BatchIterator, self).__init__(*args, **kwds)
        self.shuffle = shuffle
    def __iter__(self):
        if self.shuffle:
            self.X, self.y = shuffle(self.X, self.y)
        for res in super(BatchIterator, self).__iter__():
            yield res

class ImageBatchIterator(BatchIterator):
    def __init__(self, filenames, prefix, *args, **kwds):
        super(ImageBatchIterator, self).__init__(*args, **kwds)
        self.filenames = filenames
        self.prefix = prefix
    def transform(self, Xb, yb):
        filenames = np.asarray( [ self.filenames[int(x)] for x in Xb.flatten().tolist() ] )
        if self.prefix == "":
            Xb_actual = np.asarray( [ load_image(x) for x in filenames ], dtype="float32" )
        else:
            Xb_actual = np.asarray( [ load_image(self.prefix + os.path.sep + x) for x in filenames ], dtype="float32" )
        return Xb_actual, yb
    """
    def __iter__(self):
        if self.shuffle:
            self.X, self.y = shuffle(self.X, self.y)
        for res in super(ImageBatchIterator, self).__iter__():
            yield res
    """

class ReshapeBatchIterator(BatchIterator):
    def __init__(self, tp, *args, **kwds):
        super(ReshapeBatchIterator, self).__init__(*args, **kwds)
        self.tp = tp
    def transform(self, Xb, yb):
        Xb = Xb.reshape( Xb.shape[0], Xb.shape[1], self.tp[0], self.tp[1] )
        return Xb, yb

def write_stats(info, filename):
    f = open(filename, "wb")
    if "valid_accuracy" not in info[0]:
        f.write("epoch,train_loss,train_loss_best,valid_loss,valid_loss_best,dur\n")
    else:
        f.write("epoch,train_loss,train_loss_best,valid_loss,valid_loss_best,valid_accuracy,dur\n")
    for row in info:
        if "valid_accuracy" not in row:
            f.write("%f,%f,%f,%f,%f,%f\n" % (row["epoch"], row["train_loss"], \
                row["train_loss_best"], row["valid_loss"], row["valid_loss_best"], row["dur"]))
        else:
            f.write("%f,%f,%f,%f,%f,%f,%f\n" % (row["epoch"], row["train_loss"], \
                row["train_loss_best"], row["valid_loss"], row["valid_loss_best"], row["valid_accuracy"], row["dur"]))    
    f.close()

def save_stats_at_every(schedule, filename):
    def after_epoch(net, info):
        if info[-1]["epoch"] % schedule == 0:
            write_stats(info, filename)
    return after_epoch

def abs_error(a,b):
    return T.abs_(a-b)
