##############################
# Image Utility Functions    #
# Author: Garwah Lam         #
# Email: hello@garwahlam.com #
##############################

import cv2
import numpy as np

def imfill(image):
    """
    TODO: docstring
    Image is assumed to be thresholded already
    """
    start = (0,0)
    color = 255

    # Create a mask used to flood fill
    flooded = image.copy()
    height, width = image.shape[:2]
    mask = np.zeros((height+2,width+2),np.uint8)

    # Floodfill and invert
    cv2.floodFill(flooded, mask, start, color)
    inverse_floodfill = cv2.bitwise_not(flooded)

    # Return combination of image and inverted floodfill
    return cv2.bitwise_or(image,inverse_floodfill)



