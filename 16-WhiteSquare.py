import cv2
import numpy as np

img1 = np.zeros((400,600),dtype=np.uint8)
img1[100:300,200:400] = 90
cv2.imshow('Imagenartificial', img1)

img2 = np.zeros((400,600), dtype=np.uint8)
img2 = cv2.circle(img2,(300,200),125,(255),-1)
cv2.imshow('Imagenartificial2', img2)

img3 = cv2.bitwise_or(img1, img2)

img4 = cv2.bitwise_xor(img3, img2)
cv2.imshow('imagenartificial3', img4)
#cv2.imwrite('Cuadritoblancoxd.jpg', img1)

cv2.waitKey(0)
cv2.destroyAllWindows()