######################################
# Automatic License Plate Recognizer #
# Author: Garwah Lam                 #
# Email: hello@garwahlam.com         #
######################################

import argparse
import cv2
import imutils
import mahotas
import numpy as np

# Setup
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True)
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

median_kernal = 5
average_kernal = (20,20) 
opening_kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(71,71)) 
x_deriv = 1
y_deriv = 0

############################
# License Plate Extraction #
############################

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#median = cv2.medianBlur(image, median_kernal)
#blurred = cv2.blur(median, average_kernal)
blurred = cv2.bilateralFilter(image,5,21,21)
# Adaptive Histogram Equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
contrast = clahe.apply(blurred)
opening = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, opening_kernal)
intensity_diff = contrast - opening

cv2.imshow("Original",image)
cv2.imshow("Blurred", blurred)
cv2.imshow("Contrast Enhanced", contrast)
cv2.imshow("Opening", opening)
cv2.imshow("Intensity diff", intensity_diff)

# Binarization with Otsu's method
threshold = mahotas.thresholding.otsu(intensity_diff)
binary = intensity_diff.copy()
binary[binary > threshold] = 255
binary[binary < threshold] = 0

cv2.imshow("Binary", binary)

# Sobel Filter
sobelX = cv2.Sobel(binary, cv2.CV_64F, x_deriv, y_deriv)
sobelY = cv2.Sobel(binary, cv2.CV_64F, y_deriv, x_deriv)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobel = cv2.bitwise_or(sobelX, sobelY)

cv2.imshow("Sobel", sobel)

kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
dilation = cv2.dilate(sobel, kernal, iterations = 1)

cv2.imshow("Dilation", dilation)

cv2.imshow("Flooded", imutils.imfill(dilation))

cv2.waitKey(0)

##############################
# License Plate Segmentation #
##############################

#########################
# Character recognition #
#########################
