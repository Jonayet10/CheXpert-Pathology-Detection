import cv2
import numpy as numpy
import pandas as pd

def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

img = cv2.imread('view1_frontal.jpg')
out = cv2.addWeighted(img, 2, img, 0, 0)
img_new = cv2.blur(out, (5,5))
cv2.imshow("original", img_new)
cv2.waitKey(0)
