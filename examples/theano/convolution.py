"""
Simple implementation using convolution.
AUTHOR: ASHWINKUMAR GANESAN.
"""
%matplotlib inline

import theano
from theano import tensor as T
from theano.tensor.nnet import conv

import pylab
from PIL import Image
from cairocffi import *