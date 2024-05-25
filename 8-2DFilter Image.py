import cv2
import numpy as np 
img= cv2.imread('img/flor.jpg')
rows, cols = img.shape[:2]
kernel_identy = np.array([[0,0,0],[0,1,0],[0,0,0]])
kernel_3x3 = np.ones((3,3), np.float32) / 9.0
kernel_5x5 = np.ones((5,5), np.float32) / 25.0
cv2.imshow('original', img)
output = cv2.filter2D(img, -1, kernel_identy)
cv2.imshow('identity filter', output)
output = cv2.filter2D(img, -1, kernel_3x3)
cv2.imshow('3x3 filter', output)
output = cv2.filter2D(img, -1, kernel_5x5)
cv2.imshow('5x5 filter', output)
cv2.waitKey(0)