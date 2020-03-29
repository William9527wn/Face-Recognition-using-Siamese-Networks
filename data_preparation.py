# Helper functions to arrange the dataset into the format required for training the 
# siamese networks

# Importing the libraries

import re
import numpy as np
from PIL import Image

def read_image(filename, byteorder='>'):
    # read raw file to buffer
    with open(filename,'rb') as f:
        buffer = f.read()
        # We extract the header, width, height and maxvalue of the image
        header, width, height, maxvalue = re.search(b"(^P5\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        # Convert image to numpy array using np.frombuffer 
        image_numpy = np.frombuffer(buffer, dtype='u1' if int(maxvalue)<256 else byteorder+'u2',
                                    count = int(width)* int(height),
                                    offset = len(header)
                                    ).reshape(int(height), int(width))
        return image_numpy

def test_read_image():
    img = Image.open('Data\s1\1.pgm')
    img = read_image('Data\s1\1.pgm')
    print(img.shape)
