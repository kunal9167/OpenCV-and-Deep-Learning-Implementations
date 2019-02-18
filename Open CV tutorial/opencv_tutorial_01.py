import imutils
import cv2
#
image=cv2.imread("jp.jpg")
(h,w,d) = image.shape
print("width={}, height={}, depth={}".format(w,h,d))
#
# cv2.imshow("Image",image)
# cv2.waitKey(0)
#
# # # extract a 100x100 pixel square ROI (Region of Interest) from the
# # # input image starting at x=320,y=60 at ending at x=420,y=160
# # roi = image[60:160, 320:420]
# # cv2.imshow("ROI", roi)
# # cv2.waitKey(0)
# #
# # resized=cv2.resize(image,(600,600))
# # cv2.imshow("Resized",resized)
# # cv2.waitKey(0)
# #
# # # Considering aspect Ratio
# r=300/w
# dim = (300,int(h*r))
# resized = cv2.resize(image,dim)
# cv2.imshow("Resized Image with aspect ratio", resized)
# cv2.waitKey(0)
#
#
# # Using Imutils to resize
# resized = imutils.resize(image,width=300)
# cv2.imshow("Imutils Resize",resized)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()
# #Rotating the image
# center = (w//2,h//2)
# M=cv2.getRotationMatrix2D(center,45,1.0)
# rotated = cv2.warpAffine(image,M,(w,h))
# cv2.imshow("Rotated Image using openCV",rotated)
# cv2.waitKey(0)
#
# # Rotate using Imutils
# rotated = imutils.rotate(image,45)
# cv2.imshow("Imutils Rotation", rotated)
# cv2.waitKey(0)
#
# # Rotate image with bound
# rotated=imutils.rotate_bound(image,-45)
# cv2.imshow("Bounded Rotation",rotated)
# cv2.waitKey(0)


cv2.destroyAllWindows()

# # Blur an image
# blur = cv2.GaussianBlur(image,(11,11),0)
# cv2.imshow("Blurred Image",blur)
# cv2.waitKey(0)
#
# # Drawing a rectangle on the image
# output = image.copy()
# cv2.rectangle(output,(320,60), (410,180), (0,0,255), 2)
# cv2.imshow("Rectangle",output)
# cv2.waitKey(0)
#
# # Draw a circle on the image
# output = image.copy()
# cv2.circle(output,(300,150),20,(255,0,0),-1)
# cv2.imshow("Circle", output)
# cv2.waitKey(0)
#
# # Draw a line on the image
# output = image.copy()
# cv2.line(output,(60,20),(400,200),(0,0,255),2)
# cv2.imshow("Line", output)
# cv2.waitKey(0)

# Put text on image
output = image.copy()
cv2.putText(output,"openCV + Jurassic Park", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
cv2.imshow("Text",output)
cv2.waitKey(0)
