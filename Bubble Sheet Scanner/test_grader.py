import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import cv2

# Argument Parser
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True)
args=vars(ap.parse_args())

ANSWER_KEY = {0: 2, 1: 4, 2: 0, 3: 2, 4: 1}

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(blurred,75,200)

# cv2.imshow("Edged", edged)
# cv2.waitKey(0)

cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = 0

if len(cnts)>0:
	cnts = sorted(cnts,key=cv2.contourArea,reverse=True)

	for c in cnts:
		peri = cv2.arcLength(c,True)
		approx = cv2.approxPolyDP(c,0.02*peri,True)

		if(len(approx)==4):
			docCnt = approx
			break

# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# Thresholding the image
thresh = cv2.threshold(warped, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cv2.imshow("Edged", thresh)
# cv2.waitKey(0)

# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts:
	# Computing bounding box
	(x,y,w,h) = cv2.boundingRect(c)
	ar = w/float(h)

	# in order to label the contour as a question, region
	# should be sufficiently wide, sufficiently tall, and
	# have an aspect ratio approximately equal to 1
	if w>=20 and h>=20 and ar>=0.9 and ar<=1.1:
		questionCnts.append(c)

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionCnts = contours.sort_contours(questionCnts,method="top-to-bottom")[0]
correct = 0

for (q,i) in enumerate(np.arange(0,len(questionCnts),5)):
	cnts=contours.sort_contours(questionCnts[i:i+5])[0]
	bubbled = None

	for (j,c) in enumerate(cnts):
		mask = np.zeros(thresh.shape,dtype="uint8")
		cv2.drawContours(mask,[c],-1,255,-1)

		# apply the mask to the thresholded image, then
		# count the number of non-zero pixels in the
		# bubble area
		mask = cv2.bitwise_and(thresh,thresh,mask=mask)
		total = cv2.countNonZero(mask)

		# if the current total has a larger number of total
		# non-zero pixels, then we are examining the currently
		# bubbled-in answer
		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)

		# cv2.imshow("Bubble", mask)
		# cv2.waitKey(0)

	# initialize the contour color and the index of the
	# *correct* answer
	color = (0, 0, 255)
	k = ANSWER_KEY[q]

	# check to see if the bubbled answer is correct
	if k == bubbled[1]:
		color = (0, 255, 0)
		correct += 1

	# draw the outline of the correct answer on the test
	cv2.drawContours(paper, [cnts[k]], -1, color, 3)

# grab the test taker
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)
