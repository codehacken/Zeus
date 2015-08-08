# This is an example program to construct a simple logistic regression using
# Theano.

#!/usr/bin/python

import pydot
import theano as th                             # This is the theano library.
import theano.tensor as T                       # Tensor from theano.
from numpy import random as rn                  # Random number generator.


# Define the inputs.
N = 400                                         # This is the number of input data points.
feats = 784                                     # This is the number of the features / data point.

# Generate the basic input training data set.
# The low is 0 and high is 2 but not included.
D = (rn.randn(N, feats), rn.randint(size=N, low=0, high=2))
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
x = T.dmatrix('x')
y = T.vector('y')

# Initialize the weight vectors.
w = th.shared(rn.randn(feats), name="w")
b = th.shared(0., name="b")

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))         # Probability that target = 1
prediction = p_1 > 0.5                          # The prediction thresholded.
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)   # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()      # The cost to minimize
gw, gb = T.grad(cost, [w, b])                   # Compute the gradient of the cost
                                                # (we shall return to this in a
                                                # following section of this tutorial)

# Compile
# 0.1 is the learning on the gradient.
train = th.function(
    inputs=[x, y],
    outputs=[prediction, xent],
    updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = th.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print "Final model:"
print w.get_value(), b.get_value()
print "target values for D:", D[1]
print "prediction on D:", predict(D[0])

# Generate the training graph.
# There is th.printing.pprint and th.printing.debugprint too.
th.printing.pydotprint(train, outfile="graphs/train.png", var_with_name_simple=True)
