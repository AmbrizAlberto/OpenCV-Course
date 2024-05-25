import cv2

# Leer la imagen
img = cv2.imread('img/mariposa.jpg')

# Cambiar el tamaño de la imagen original a una escala de x=0.5 y=0.5
# indicando que se reduzca la mitad
img2 = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

# Mostrar la imagen original
cv2.imshow('mariposa', img)

# Mostrar la imagen reducida
cv2.imshow('mariposa', img2)

# Guardar la imagen reducida
cv2.imwrite("img/mariposa_reducida.jpg", img2)

# Cerrar las pestañas
cv2.waitKey()