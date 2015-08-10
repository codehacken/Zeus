# This is an example program to construct a simple logistic regression using
# Theano.

#!/usr/bin/python

from __future__ import print_function
import theano                                   # This is the theano library.
import theano.tensor as tt                      # Tensor from theano.
from numpy import random as rn                  # Random number generator.
import numpy as np

theano.config.floatX = 'float32'

# Define the inputs.
N = 400                                         # This is the number of input data points.
feats = 784                                     # This is the number of the features / data point.

# Generate the basic input training data set.
# The low is 0 and high is 2 but not included.
D = (rn.randn(N, feats).astype(theano.config.floatX),
     rn.randint(size=N, low=0, high=2).astype(theano.config.floatX))
training_steps = 10000

'''
Understanding the types of variables and how to access the information.

                    >>> type(y.owner.inputs[1])
                    <class 'theano.tensor.var.TensorVariable'>
                    >>> type(y.owner.inputs[1].owner)
                    <class 'theano.gof.graph.Apply'>
                    >>> y.owner.inputs[1].owner.op
                    <theano.tensor.elemwise.DimShuffle object at 0x106fcaf10>
                    >>> y.owner.inputs[1].owner.inputs
                    [TensorConstant{2.0}]
'''

# Construct the graph.
# Declare all the THEANO variables.
x = tt.matrix('x')
y = tt.vector('y')

# Initialize the weight vectors.
w = theano.shared(rn.randn(feats).astype(theano.config.floatX), name="w")
b = theano.shared(np.asarray(0., dtype=theano.config.floatX), name="b")

x.tag.test_value = D[0]
y.tag.test_value = D[1]

# Construct Theano expression graph
p_1 = 1 / (1 + tt.exp(-tt.dot(x, w) - b))         # Probability that target = 1
prediction = p_1 > 0.5                          # The prediction thresholded.
xent = -y * tt.log(p_1) - (1-y) * tt.log(1-p_1)   # Cross-entropy loss function
cost = tt.cast(xent.mean(), 'float32') + \
       0.01 * (w ** 2).sum()                    # The cost to minimize
gw, gb = tt.grad(cost, [w, b])                   # Compute the gradient of the cost
                                                # (we shall return to this in a
                                                # following section of this tutorial)

# Compile
# 0.1 is the learning on the gradient.
train = theano.function(
    inputs=[x, y],
    outputs=[prediction, xent],
    updates={w: w - 0.01 * gw, b: b - 0.01 * gb},
    name="train")

predict = theano.function(inputs=[x], outputs=prediction, name="predict")

if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
        train.maker.fgraph.toposort()]):
    print('Used the cpu')
elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
          train.maker.fgraph.toposort()]):
    print('Used the gpu')
else:
    print('ERROR, not able to tell if theano used the cpu or the gpu')
    print(train.maker.fgraph.toposort())

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

# print("Final model:")
# print(w.get_value(), b.get_value())
print("target values for D:", D[1])
print("prediction on D:", predict(D[0]))

# Generate the training graph.
# There is th.printing.pprint and th.printing.debugprint too.
theano.printing.pydotprint(train, outfile="graphs/train.png", var_with_name_simple=True)
