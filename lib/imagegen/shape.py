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

# The fill color and outline color are parameters in the base
# shape class.
class _Shape(object):
    # Define the class as abstract.
    __metaclass__ = ABCMeta
    
    # Define the number of vertices in the shape
    # Property: how the vertices generate edges.
    #           E.g. - The edges are equal.
    def __init__(self, n_vertex, **kwargs):
        def assert_error_message(param):
            return "Parameter " + param + " is missing."
        
        # Check if fill_color exists.
        assert 'fill_color' in kwargs, assert_error_message('fill_color')
        self.fill_color = kwargs['fill_color']
        
        # Check if outline_color exists.
        assert 'outline_color' in kwargs, assert_error_message('outline_color')
        self.outline_color = kwargs['outline_color']

        # Defining other parameters.
        self.ndims = n_vertex
        
        # Define the vertices.
        self.vertices = {}
        
    # Generate the vertices of the shape given a initial vertex
    # and length of edges (if necessary).
    @abstractmethod
    def _generate_vertices(self):
        pass
    
    # Helper Functions.
    def get_vertices(self):
        vertices = {}
        for vertex in self.vertices:
            vertices[vertex] = self.vertices[vertex].get_pos()
        
        return vertices
    """
    Generate the image_dr so that the figure can be placed
    inside the image at specific place.
    """
    @abstractmethod
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

The vertices are index with their relative positions.
lu: Left upper corner.
ld: Left lower corner.
ru: Right upper corner.
rd: Right lower corner.
"""

class Square(_Shape):
    # Constructor with color, vertex and length as ndims is constant @ 4.
    def __init__(self, initial_vertex, edge_length, **kwargs):   
        # The outline of the square and fill color are the same.
        super(Square, self).__init__(4, **kwargs)
        self._generate_vertices(initial_vertex, edge_length)
        self.edge_length = edge_length
    
    # Vertices are generated as A, B, C, D
    def _generate_vertices(self, initial_vertex, edge_length):
        self.vertices = {'lu': _Vertex(initial_vertex[0], initial_vertex[1]),
                         'ld': _Vertex(initial_vertex[0] + edge_length, initial_vertex[1]),
                         'ru': _Vertex(initial_vertex[0], initial_vertex[1] + edge_length),
                         'rd': _Vertex(initial_vertex[0] + edge_length, 
                          initial_vertex[1] + edge_length),
                        }
     
    # Updated image_dr object with the square in it.
    def dr_add(self, image_dr):
        image_dr.rectangle((self.vertices['lu'].get_pos(),
                           self.vertices['rd'].get_pos()), fill = self.fill_color,
                           outline = self.outline_color)

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
    def __init__(self, center, diameter, **kwargs):
        # The outline of the circle and fill color are the same.
        
        # The ndims for a circle is defined as 1.
        # This is because it has infinite vertices, so the number of points is only defined
        # the center of the circle.
        super(Circle, self).__init__(1, **kwargs)
        self._generate_vertices(center)
        self.diameter = diameter
    
    # For a circle only center's position is stored in the vertices.
    def _generate_vertices(self, center):
        self.vertices = {'center': _Vertex(center[0], center[1])}
    
    # Updated image_dr object with the circle in it.
    
    # For drawing the circle, we create a bounding box in which
    # it is drawn, defined by the position (center - r, center + r) for both
    # x & y. 
    def dr_add(self, image_dr):
        center = self.vertices['center'].get_pos()
        print(center)
        r = self.diameter / 2
        image_dr.ellipse(((center[0] - r, center[1] - r),
                          (center[0] + r, center[1] + r)),
                         fill = self.fill_color, 
                         outline = self.outline_color)

"""
Define a Triangle.
The Triangle is equilateral and defined by 3 parameters:
1. The top vertex of the triangle.
2. The length of the side.
3. The fill_color & outline color is used to define the color to be filled
   in the circle and the boundary respectively.
It has three points in a standard configuration i.e. top, left and right.

TODO: Add triangle rotation of points to get different pictures.
"""
class Triangle(_Shape):
    # Constructor with color, top vertex of the Triangle & length of the side.
    def __init__(self, top_vertex, side_length, **kwargs):
        # The outline of the triangle and fill color are the same.
        
        # The ndims for a triangle is defined as 3.
        # Any one side length is required as it is a equilateral triangle.
        super(Triangle, self).__init__(3, **kwargs)
        self._generate_vertices(top_vertex, side_length)
        self.side_length = side_length
    
    # There are 3 points in the triangle that are stored as:
    # t: the top vertex.
    # lb: the left bottom vertex.
    # rb: the right bottom vertex.
    def _generate_vertices(self, top_vertex, side_length):
        lcos30 = (side_length * 2) / 3
        lsin30 = (side_length / 2)
        self.vertices = {'t': _Vertex(top_vertex[0], top_vertex[1]),
                         'lb': _Vertex(top_vertex[0] - lsin30, top_vertex[1] + lcos30),
                         'rb': _Vertex(top_vertex[0] + lsin30, top_vertex[1] + lcos30)}

    # Updated image_dr object with the triangle in it.
    # Use the polygon function to draw a triangle and will it with a color. 
    def dr_add(self, image_dr):
        # Get the points.
        t = self.vertices['t'].get_pos()
        lb = self.vertices['lb'].get_pos()
        rb = self.vertices['rb'].get_pos()
        
        image_dr.polygon((t, lb, rb),
                         fill = self.fill_color, 
                         outline = self.outline_color)
