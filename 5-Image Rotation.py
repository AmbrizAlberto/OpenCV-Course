import cv2
import numpy as np

# Dezplaza la imagen a la izquierda y hacia abajo
img = cv2.imread('img/mariposa.jpg')

num_rows, num_cols = img.shape[:2]
translation_matrix = np.float32([ [1,0,70], [0,1,110] ])
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

cv2.imshow('Translation', img_translation)
cv2.waitKey()

# Hace una rotacion a la imagen
img2 = cv2.imread('img/mariposa.jpg')

num_rows, num_cols = img2.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2),30, 1)
img_rotation2 = cv2.warpAffine(img2, rotation_matrix, (num_cols, num_rows))

cv2.imshow('Rotation', img_rotation2)

cv2.waitKey()