import cv2
import numpy as np

img = cv2.imread("img/flor.jpg")
texto = "Esto es una flor en invierno"
coord_texto = (130,50)
img = cv2.putText(img, texto, coord_texto, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255,255,0), thickness=1, lineType=cv2.LINE_AA)

centro = (252,182)
radio = 100
cv2.circle(img,centro,radio,(255,255,0),thickness=2, lineType=cv2.LINE_AA)
puntoA = (150, 80)
puntoB = (350, 80)
puntoC = (150, 280)
puntoD = (350, 280)
img = cv2.line(img, puntoA,puntoB,(255,255, 0),thickness=5, lineType=cv2.LINE_AA)
img = cv2.line(img, puntoA,puntoC,(255,255, 0),thickness=5, lineType=cv2.LINE_AA)
img = cv2.line(img, puntoC,puntoD,(255,255, 0),thickness=5, lineType=cv2.LINE_AA)
img = cv2.line(img, puntoD,puntoB,(255,255, 0),thickness=5, lineType=cv2.LINE_AA)

cv2.imshow("Flor original", img)
print(img.shape)

#img_crop = img[80:280, 150:330]
#cv2.imshow("imagen recortada", img_crop)

cv2.waitKey()
cv2.destroyAllWindows()