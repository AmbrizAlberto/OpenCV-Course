import cv2 
import numpy as np

img = cv2.imread('img/cubo.jpg')
img2 = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
img3 = cv2.cvtColor(img2,cv2.COLOR_BGR2YUV)

cv2.imshow('Canal R: ', img3[:,:,0])
cv2.imshow('Canal G: ', img3[:,:,1])
cv2.imshow('Canal B: ', img3[:,:,2])


cv2.imwrite("img/cubo_gris.jpg", img2)
cv2.imwrite("img/cubo_reducida.jpg", img3)
cv2.waitKey()
