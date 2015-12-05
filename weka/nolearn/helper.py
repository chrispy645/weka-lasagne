from __future__ import print_function
#import lasagne
#from lasagne import *
#from lasagne.updates import *
#from lasagne.nonlinearities import *
#from lasagne.layers import *
#from lasagne.objectives import *
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
    img = np.asarray( [ img ] )
    return img
        
def shuffle(*arrays):
    # https://github.com/dnouri/nolearn/issues/27#issuecomment-71175381
    p = np.random.permutation(len(arrays[0]))
    return [array[p] for array in arrays]

class ShufflingBatchIterator(BatchIterator):
    # https://github.com/dnouri/nolearn/issues/27#issuecomment-71175381
    def __iter__(self):
        self.X, self.y = shuffle(self.X, self.y)
        for res in super(ShufflingBatchIterator, self).__iter__():
            yield res

class ImageBatchIterator(BatchIterator):
    def __init__(self, filenames, *args, **kwds):
        super(FilenameToImageBatchIterator, self).__init__(*args, **kwds)
        self.filenames = filenames
    def transform(self, Xb, yb):
        filenames = np.asarray( [ self.filenames[int(x)] for x in Xb.flatten().tolist() ] )
        Xb_actual = np.asarray( [ load_image(x) for x in filenames ], dtype="float32" )
        return Xb_actual, yb
