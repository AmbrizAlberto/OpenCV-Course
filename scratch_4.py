import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    num_rows, num_cols = frame.shape[:2]

    translation_matrix = np.float32([ [1,0,70], [0,1,110] ])
    img_translation = cv2.warpAffine(frame, translation_matrix, (num_cols, num_rows))

    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2),30, 1)
    img_rotation2 = cv2.warpAffine(img_translation, rotation_matrix, (num_cols, num_rows))
    rotacion = cv2.warpAffine(frame, rotation_matrix, (num_cols, num_rows))

    reducido = cv2.resize(img_rotation2, (0,0), fx=0.8, fy=0.8)
    reduc = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)

    color_gris = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    general = cv2.cvtColor(reducido,cv2.COLOR_BGR2GRAY)
    if ret == True:
        cv2.imshow('Camara', general)
        cv2.imshow('Camara gris', color_gris)
        cv2.imshow('Camara reducida', reduc)
        cv2.imshow('Camara traslado', img_translation)
        cv2.imshow('Camara rotada', rotacion)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllwindows()