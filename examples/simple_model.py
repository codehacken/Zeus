#!/usr/bin/python3

"""
This program shows how to create a simple model in Theano
"""

import theano as th
from theano import tensor as T
import numpy as np
import pickle

# Generate some data.
# Input Data.
trX = np.linspace(-1, 1, 101)
# Output Data.
trY = 2 * trX + np.random.randn(trX.shape[0]) * 0.33

# Once the data has been defined, declare the variables.
X = T.scalar()
Y = T.scalar()

# Define the model.
def model(X, w):
        y = X * w
        return y

# Create model parameters (the weight vector).
w = th.shared(np.asarray(0., dtype=th.config.floatX))

# Define the output.
y = model(X, w)

# Once the output is generated, define the cost function.
# Using mean square error.
cost = T.mean(T.sqr(y - Y))
# Define the gradient used to the update the weight vector.
gradient = T.grad(cost=cost, wrt=Y)
# Once the gradient is defined, update the model.
updates = [[w, w - gradient*0.01]]

# Create a shared function to train the model.
train = th.function(inputs=[X,Y], outputs=[y, cost, gradient], updates=updates, allow_input_downcast=True)

out = np.zeros([1,3], dtype='f')
for i in range(1):
        for x, y in zip(trX, trY):
                output = train(x, y)
                # Perform a horizontal stack of array values.
                out = np.vstack((out, np.hstack((output[0], output[1], output[2]))))

print(w.get_value())
# Process the output.
# for i in range(0, len(out)):
#     print(out[i][0].flatten())

# Process the output into a final list that can stored using CPickle.
with open('out.pickle', 'wb') as handle:
  pickle.dump(out, handle)