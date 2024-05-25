import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_mask = cv2.imread('img/mascara.png')
h_mask, w_mask = face_mask.shape[:2]

if face_cascade.empty():
    raise IOError('Error al cargar archivo .xml')

cap = cv2.VideoCapture(0)
scaling_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face_rects:
        if h > 0 and w > 0:
            h, w = int(1.4 * h), int(1.0 * w)
            y -= int(0.1 * h)
            frame_roi = frame[y:y + h, x:x + w]
            face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)
            gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_mask, 180, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
            frame[y:y + h, x:x + w] = cv2.add(masked_face, masked_frame)
    cv2.imshow('Cara con mascara', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()