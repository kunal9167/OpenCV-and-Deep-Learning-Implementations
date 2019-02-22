from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

ap=argparse.ArgumentParser()
ap.add_argument("-v","--video",help="path to the video file")
ap.add_argument("-b","--buffer",type=int,default=64,help="max buffer size")
args = vars(ap.parse_args())

# Define the lower and upper boundaries of the "color" ball in the hsv color space
Lower = (19, 54, 213)
Upper = (40, 165, 255)
pts=deque(maxlen=args["buffer"])

# if a video path was supplied, grab the reference
if not args.get("video"):
	vs = VideoStream(src=0).start()
else:
	vs=cv2.VideoCapture(args["video"])

time.sleep(2.0)


while True:

	frame=vs.read()
	frame = frame[1] if args.get("video",False) else frame

	#If video has ended
	if frame is None:
		break

	frame = imutils.resize(frame,width=600)
	blurred = cv2.GaussianBlur(frame,(11,11),0)
	hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(hsv,Lower,Upper)
	mask = cv2.erode(mask,None,iterations=2)
	mask = cv2.dilate(mask,None, iterations=2)

	# find contours in the mask and initialize the current (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	if(len(cnts)>0):
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c=max(cnts,key=cv2.contourArea)
		((x,y),radius) = cv2.minEnclosingCircle(c)

		M=cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		if(radius>10):
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
			cv2.circle(frame,center,5,(0,0,255),-1)

	pts.appendleft(center)

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
