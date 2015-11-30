"""
Implement the geometric shapes to generate the images.
Using the Python Imaging Library (PIL).
"""
import numpy as np

# PIL library to process images.
import pylab
from PIL import Image, ImageFont, ImageDraw

# Create an abstract class.
from abc import ABCMeta, abstractmethod

# Class for vertices. (2-D Vertex)
class _Vertex(object):
    # Define the location of the point.
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    # Helper Functions.
    def get_pos(self):
        return (self.x, self.y)

"""
Define the Shape class as abstract and hidden.
Edges are not defined as image draw will automatically be generated.
"""

# Make the abstract class hidden as it not to be imported.
class _Shape(object):
    # Define the class as abstract.
    __metaclass__ = ABCMeta
    
    # Define the number of vertices in the shape
    # Property: how the vertices generate edges.
    #           E.g. - The edges are equal.
    def __init__(self, n_vertex, **kwargs):
        assert 'fill_color' in kwargs, "Parameter fill_color is missing."
        self.fill_color = kwargs['fill_color']
        
        assert 'outline_color' in kwargs, "Parameter outline_color is missing."
        self.outline_color = kwargs['outline_color']
        
        if 'fill_color' in kwargs:
            self.fill_color 
        # Check the **kwargs.
        #for name, arg in kwargs.items():
        print(kwargs) 
        self.ndims = n_vertex
        self.fill_color = fill_color
        self.outline_color = outline_color
        
    # Generate the vertices of the shape given a initial vertex
    # and length of edges (if necessary).
    #@abstractmethod
    def _generate_vertices(self):
        pass
    
    """
    Generate the image_dr so that the figure can be placed
    inside the image at specific place.
    """
    #@abstractmethod
    def dr_add(self, image_dr):
        pass
    
# Define the Square class to generate it.
"""
Square class __init__ parameters:
1. color -- Color to be filled in the square.
2. initial_vertex -- tuple of size 2 which contains the x, y
                     co-ordinates of the left upper corner.
3. edge_length -- length of each side of the square.
A --- B
|     |
D --- C
"""

class Square(_Shape):
    # Constructor with color, vertex and length as ndims is constant @ 4.
    def __init__(self, initial_vertex, edge_length, **kwargs):   
        # The outline of the square and fill color are the same.
        super(Square, self).__init__(4, color, color)
        self._generate_vertices(initial_vertex, edge_length)
    
    # Vertices are generated as A, B, C, D
    def _generate_vertices(self, initial_vertex, edge_length):
        self.vertices = [_Vertex(initial_vertex[0], initial_vertex[1]),
                         _Vertex(initial_vertex[0] + edge_length, initial_vertex[1]),
                         _Vertex(initial_vertex[0], initial_vertex[1] + edge_length),
                         _Vertex(initial_vertex[0] + edge_length, 
                          initial_vertex[1] + edge_length),
                        ]
    
    # Helper Functions.
    def get_vertices(self):
        vertices = []
        for vertex in self.vertices:
            vertices.append(vertex.get_pos())
        
        return vertices
    
    # Updated image_dr object with the square in it.
    def dr_add(self, image_dr):
        image_dr.rectangle((self.vertices[0].get_pos(),
                           self.vertices[3].get_pos()), fill = self.fill_color,
                           outline = self.outline_color)
        return image_dr

"""
Define a Circle.
A circle is defined by 3 parameters:
1. The center of the circle.
2. The diameter of the circle.
3. The fill_color & outline color is used to define the color to be filled
   in the circle and the boundary respectively.
"""
class Circle(_Shape):
    # Constructor with color, center of the circle & diameter.
    def __init__(self, color, center, diameter):
        # The outline of the square and fill color are the same.
        
        # The ndims for a circle is defined as 1.
        # This is because it has infinite vertices, so the number of points is only defined
        # the center of the circle.
        super(Circle, self).__init__(1, color, color)
        self._generate_vertices(initial_vertex, edge_length)
    
    # Vertices are generated as A, B, C, D
    def _generate_vertices(self, initial_vertex, edge_length):
        self.vertices = [_Vertex(initial_vertex[0], initial_vertex[1]),
                         _Vertex(initial_vertex[0] + edge_length, initial_vertex[1]),
                         _Vertex(initial_vertex[0], initial_vertex[1] + edge_length),
                         _Vertex(initial_vertex[0] + edge_length, 
                          initial_vertex[1] + edge_length),
                        ]
    
    # Helper Functions.
    def get_vertices(self):
        vertices = []
        for vertex in self.vertices:
            vertices.append(vertex.get_pos())
        
        return vertices
    
    # Updated image_dr object with the square in it.
    def dr_add(self, image_dr):
        image_dr.rectangle((self.vertices[0].get_pos(),
                           self.vertices[3].get_pos()), fill = self.fill_color,
                           outline = self.outline_color)
        return image_dr
