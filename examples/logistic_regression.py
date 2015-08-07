# This is an example program to construct a simple logistic regression using
# Theano.

#!/usr/bin/python

import numpy as np
import theano as th # This is the theano library.
import theano.tensor as T # Tensor from theano.
from numpy import random as rn # Random number generator.


# Define the inputs.
N = 400  # This is the number of input data points.
feats = 784  # This is the number of the features / data point.

# Generate the basic input training data set.
# The low is 0 and high is 2 but not included.
D = (rn.randn(N, feats), rn.randint(size=N, low=0, high=2))
training_steps = 10000

# Construct the graph.
# Declare all the theano variables.


print D


