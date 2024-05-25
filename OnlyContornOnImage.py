import cv2
import numpy as np

valorGauss=3
valorKernel=3

original=cv2.imread('img/mariposa.jpg')
gris=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
gauss=cv2.GaussianBlur(gris, (valorGauss, valorKernel), 0 )
#Mostrar resultados

cv2.imshow("Grises",gris)
cv2.imshow("Gauss",gauss)
cv2.waitKey(0)