import cv2 

# Imprime los codigos de los colores
print([x for x in dir(cv2) if x.startswith('COLOR_')])

img = cv2.imread('img/cubo.jpg')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img3 = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

cv2.imshow('cubo en gris', img2)
cv2.imshow('cubo reducido',img3)

cv2.imwrite("img/cubo_gris.jpg", img2)
cv2.imwrite("img/cubo_reducida.jpg", img3)

cv2.waitKey()

""" CODIGO PARA GUARDAR LA IMAGEN CUBO.JPG EN TODOS LOS FORMATOS EXISTENTES
contador = 0
img = cv2.imread("img/cubo.jpg")
for x in dir(cv2): 
    if x.startswith('COLOR_'):
        print("cv2." + x + ",")
        try:
            img2 = cv2.cvtColor(img, getattr(cv2, x))
            cv2.imwrite('CUBOCOLOR_{}'.format(contador), img2)
        except:
            print("Error en el color")

cv2.waitKey() 
"""