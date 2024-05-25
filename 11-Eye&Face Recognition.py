import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
leftear_cascade = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
rightear_cascade = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')
boca_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nariz_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

if face_cascade.empty():
    raise IOError('No se puedo cargar el filtro de cara')
if eye_cascade.empty():
    raise IOError('No se puede cargar el filtro de ojo')

cap = cv2.VideoCapture(0)
if(cap.isOpened() == False):
    print("Error al leer la camara!!!!")
ds_factor = 1.2

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    leftear = leftear_cascade.detectMultiScale(gray, 1.3,5)
    rightear = rightear_cascade.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in leftear:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),3)
        cv2.putText(frame, "Oreja Derecha", (x,y), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255,255,0), thickness=1, lineType=cv2.LINE_AA)

    for(x,y,w,h) in rightear:
        cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),3)
        cv2.putText(frame, "Oreja Izquierda", (x,y), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255,255,0), thickness=1, lineType=cv2.LINE_AA)


    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 2, 5)
        bocaxd = boca_cascade.detectMultiScale(gray,2,5)
        nariz = nariz_cascade.detectMultiScale(gray,2,5)
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            cv2.putText(frame, "Ojo", (x+x_eye+int(w_eye*0.2),y+y_eye+int(h_eye*0.5)), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255,255,0), thickness=1, lineType=cv2.LINE_AA)
            radius = int(0.3*(w_eye + h_eye))
            color = (0,255,0)
            thickness = 3
            cv2.circle(roi_color, center, radius, color, thickness)

        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, "Al DEV", (x,y), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255,255,0), thickness=1, lineType=cv2.LINE_AA)
        for(x,y,w,h) in bocaxd:
            cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),3)
            cv2.putText(frame, "boca", (x,y), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255,255,0), thickness=1, lineType=cv2.LINE_AA)
            break
        for(x,y,w,h) in nariz:
            cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),3)
            cv2.putText(frame, "nariz", (x,y), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255,255,0), thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow("DETECTOR DE ROSTROS Y OJOS", frame)
    c = cv2.waitKey(1)
    if c==27:
        break
cap.release()
cv2.destroyAllWindows()