######################################
# Automatic License Plate Recognizer #
# Author: Garwah Lam                 #
# Email: hello@garwahlam.com         #
######################################

import argparse
import cv2
import mahotas
import numpy as np

# Setup
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True)
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

median_kernal  = 5
average_kernal = (20,20) 
x_deriv = 1
y_deriv = 0

############################
# License Plate Extraction #
############################

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(image.copy(), median_kernal)
blurred = cv2.blur(blurred, average_kernal)
intensity_diff = image - blurred

# Binarization with Otsu's method
threshold = mahotas.thresholding.otsu(intensity_diff)
# Sobel Filter
cv2.imshow("Original",image)
cv2.imshow("Blurred", blurred)
cv2.imshow("Thresholded", thresh)

cv2.waitKey(0)

##############################
# License Plate Segmentation #
##############################

#########################
# Character recognition #
#########################
