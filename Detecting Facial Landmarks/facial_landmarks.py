# Import the package
# from imutils import face_utils
import rect_to_bb
import numpy as np
import argparse
import imutils
import dlib
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-p","--shape",required=True,help="path to facial landmark")
ap.add_argument("-i","--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# initialise dlib's face detector and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape"])

image = cv2.imread(args["image"])
image = imutils.resize(image,width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray,1)

# loop over the face detections
for (i,rect) in enumerate(rects):
    shape=predictor(gray,rect)
    shape=rect_to_bb.shape_to_np(shape)
    print(rect.left())
    (x,y,w,h) = rect_to_bb.rect_to_bb(rect)

    #Creating a rectangle around the face
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    #Show the face number
    cv2.putText(image,"Face {}".format(i+1),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    for (x,y) in shape:
        cv2.circle(image,(x,y),1,(0,0,255),-1)

cv2.imshow("Output",image)
cv2.waitKey(0)
