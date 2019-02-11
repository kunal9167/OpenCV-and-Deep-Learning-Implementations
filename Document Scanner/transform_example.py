from transform import four_point_transform
import numpy as np
import argparse
import cv2

# Construct the arguement parse and parse the arguements
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image", help='path to the image')
ap.add_argument("-c","--coords", help='comma seperated list of source points')
args= vars(ap.parse_args())

# Load the image and grab the source coordinates i.e the list of (x,y) points
image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]),dtype='float32')

# Apply the four point transform to obtain a "birds eye view" of the image
warped = four_point_transform(image,pts)

cv2.imshow("Original",image)
cv2.imshow("Warped",warped)
cv2.waitKey(0)
