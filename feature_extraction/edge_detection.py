# from https://learnopencv.com/edge-detection-using-opencv/

import cv2

img = cv2.imread('view1_frontal.jpg')

cv2.imshow('Original', img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

edges = cv2.Canny(image = img_gray, threshold1 = 100, threshold2 = 150)

cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
