#!/usr/bin/python3
"""
This is to create a simple function in Theano.
"""

# Declare the theano library.
import theano as th
# Import tensors.
from theano import tensor as T

# Declare scalar variables that use to compute.
a = T.scalar()
b = T.scalar()

# Define the operation.
y = a * b

# Create the theano function to compute the operation.
multiply = th.function(inputs=[a, b], outputs=y)

print(multiply(1, 2))
print(multiply(3, 3))

