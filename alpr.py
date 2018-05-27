######################################
# Automatic License Plate Recognizer #
# Author: Garwah Lam                 #
# Email: hello@garwahlam.com         #
######################################

import numpy as np
import cv2
import argparse

# Setup
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True)
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

median_kernal  = 5
average_kernal = 20 

# License Plate Extraction
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.medianBlur(image, median_kernal)
image = cv2.blur(image, average_kernal)

# License Plate Segmentation


# Character recognition
