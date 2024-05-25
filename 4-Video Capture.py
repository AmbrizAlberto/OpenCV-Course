import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
    print("Error al leer la camara!!!")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('video/video.avi', cv2.VideoWriter_fourcc('M','J','P','G'),24,(frame_width, frame_height))

while(True):
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)
        cv2.imshow('Cuadro', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()