import cv2
import numpy as np

left_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

if left_ear_cascade.empty():
    raise IOError("No se pudo cargar el archivo xml.")
if right_ear_cascade.empty():
    raise IOError("No se pudo cargar el archivo xml.")
if mouth_cascade.empty():
    raise IOError("No se pudo cargar el archivo xml.")
if nose_cascade.empty():
    raise IOError("No se pudo cargar el archivo xml.")

img = cv2.imread('img/foto.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

left_ear = left_ear_cascade.detectMultiScale(gray, 1.3, 5)
right_ear = right_ear_cascade.detectMultiScale(gray, 1.3, 5)
mouth = mouth_cascade.detectMultiScale(gray, 1.3, 5)
nose = mouth_cascade.detectMultiScale(gray, 1.3, 5)

for ( x,y,w,h) in left_ear:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

for ( x,y,w,h) in right_ear:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

for ( x,y,w,h) in mouth:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

for ( x,y,w,h) in nose:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

cv2.imshow('deteccion de orejas', img)
cv2.waitKey()
cv2.destroyAllWindows()