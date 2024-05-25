import cv2
import numpy as np
    
cap = cv2.VideoCapture(0)
valor = 'bgr'
while True:
    ret, frame = cap.read()
    
    if valor == 'gray':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if valor == 'yuv':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    if valor == 'hsv':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    key = cv2.waitKey(1) & 0xFF
    if ret == True:

        cv2.imshow('Camara', frame)
        if key == ord('g'):
            valor = 'gray'
        if key == ord('y'):
            valor = 'yuv'
        if key == ord('h'):
            valor = 'hsv'
        if key == ord('n'):
            valor = 'bgr'
        if key == ord('q') or key == ord('x'):
            break

cap.release()
cv2.destroyAllWindows()