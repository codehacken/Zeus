"""
Draw pictures containing various shapes. Contains methods to generate
jpegs with various shapes and stacks of shapes.
"""

from PIL import Image, ImageFont, ImageDraw
from lib.imagegen.shape import Square, Circle, Triangle

# Functions to generate images.
def gen_one_shape_image(image_size, max_shape_size, b_color, fill_color, outline_color):
    # Create the Image.
    im = ImageCreate(image_size, 1, background_color=b_color)
    
    # Generate squares.
    for i in range(0, image_size[0], max_shape_size):
        for j in range(0, image_size[1], max_shape_size):
            sq = Square((100, 200), 100, fill_color='black', outline_color='black')
            im.create_new_image()
            im.dr_add(sq)
    
    
    
# Draw various kinds of images.
# Creates the images.
class ImageCreate(object):
    shape_list = [Square, Circle, Triangle]
    # Define the picture size to be generated & the number of different shapes
    # in the picture.
    
    # pic_size is a tuple of size 2 (containing the NxN number of pixels).
    def __init__(self, pic_size, num_shapes, **kwargs):
        assert pic_size is not tuple, "pic_size is a tuple of size 2 (NxN)"
        self.pic_size = pic_size
        self.num_shapes = num_shapes
        
        # Check for additional parameters.
        if 'background_color' in kwargs:
            self.bcolor = kwargs['background_color']
        else:
            # Standard background color is white.
            self.bcolor = (255, 255, 255)
        
        self._im = None
        self._dr = None
        
    # Helper functions.
    # Get image.
    def get_image(self):
        return self._im
    
    # Create a new image.
    def create_new_image(self):
        self._im = Image.new('RGB', self.pic_size, self.bcolor)
        self._dr = ImageDraw.Draw(self._im)
        
        # List of shapes that the ImageDrawer takes.
        self._shape_list = []
    
    # Add shapes to the dr.
    def dr_add(self, shape):
        # Checking if the input is of some type of a shape.
        for a_shape in ImageCreate.shape_list:
            assert shape is not a_shape, "object should be of type " + a_shape
        
        shape.dr_add(self._dr)
    
        # Add the shape to the list.
        self._shape_list.append(shape)

        