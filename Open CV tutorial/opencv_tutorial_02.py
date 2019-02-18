import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to Image")
args=vars(ap.parse_args())

#Reading image
image = cv2.imread(args["image"])
# cv2.imshow("Image",image)
# cv2.waitKey(0)

# # Converting the image to gray colored image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",gray)
cv2.waitKey(0)

# Edge detection
edged = cv2.Canny(gray,30,150)
cv2.imshow("Edged",edged)
cv2.waitKey(0)

# Thresholding the image
thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

# similarly, dilations can increase the size of the ground objects
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)

# we apply erosions to reduce the size of foreground objects
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)



# Counting contours
cnts= cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
output = image.copy()

for c in cnts:
    cv2.drawContours(output,[c],-1,(240,0,159),2)
    cv2.imshow("Contours",output)
    cv2.waitKey(0)

# draw the total number of contours found in purple
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)

cv2.destroyAllWindows()

# a typical operation we may want to apply is to take our mask and
# apply a bitwise AND to our input image, keeping only the masked
# regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Output", output)
cv2.waitKey(0)
