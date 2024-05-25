import cv2
import numpy as np
import os

face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
leftear_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_mcs_leftear.xml')
rightear_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_mcs_rightear.xml')
mouth_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_mcs_mouth.xml')
nose_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_mcs_nose.xml')

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
leftear_cascade = cv2.CascadeClassifier(leftear_cascade_path)
rightear_cascade = cv2.CascadeClassifier(rightear_cascade_path)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
nose_cascade = cv2.CascadeClassifier(nose_cascade_path)

if leftear_cascade.empty():
    raise IOError("No se pudo cargar el archivo xml de izquierdo oreja")
if rightear_cascade.empty():
    raise IOError("No se pudo cargar el archivo xml de derecho oreja")
if face_cascade.empty():
    raise IOError('No se pudo abrir el archivo xml de la cara')
if eye_cascade.empty():
    raise IOError('No se pudo abrir el archivo xml de los ojos')
if nose_cascade.empty():
    raise IOError("No se pudo cargar el archivo xml de la nariz.")
if mouth_cascade.empty():
    raise IOError("No se pudo cargar el archivo xml de la boca.")

cap = cv2.VideoCapture(0)
ds_factor = 1

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    leftear = leftear_cascade.detectMultiScale(gray, 1.1 ,5)
    rightear = rightear_cascade.detectMultiScale(gray,1.1,5)
    mouth = mouth_cascade.detectMultiScale(gray, 1.3, 5)
    nose = mouth_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in leftear:
        cv2.putText(frame, "oreja", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),3)
        break
        
    for(x,y,w,h) in rightear:
        cv2.putText(frame, "oreja", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),3)
        break

    for ( x,y,w,h) in mouth:
        cv2.putText(frame, "boca", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        break

    for ( x,y,w,h) in nose:
        cv2.putText(frame, "nariz", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        break

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h//2, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,5)

        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3*(w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 3
            cv2.putText(frame, "ojo", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.circle(roi_color, center, radius, color, thickness)
        
        cv2.putText(frame, "rostro", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    cv2.imshow('Detector de rostros, ojos, nariz, boca y orejas', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()