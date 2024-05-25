import cv2
import numpy as np

img = cv2.imread("img/test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_2 = cv2.GaussianBlur(img, (3,3), 0)
sobelx = cv2.Sobel(src=img_2, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(src=img_2, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
sobelxy = cv2.Sobel(src=img_2, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

bordes = cv2.Canny(image=img_2, threshold1=100, threshold2=200)

#cv2.imshow("Difuminado Gaussiano", bordes)
#cv2.waitKey(0)

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame,(3,3), 0)
    video = cv2.Canny(image=frame, threshold1=100, threshold2=200)
    if ret == True:
        cv2.imshow('Camara', video)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllwindows()