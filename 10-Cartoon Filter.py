import cv2 
import numpy as np

def cartoonize_image(img, ds_factor=4, sketch_mode=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 7)
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100 , 255, cv2.THRESH_BINARY_INV)
    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
    num_repetitions = 10
    sigma_color = 5
    sigma_space = 7
    size = 5
    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, size, sigma_color,sigma_space)
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR)
    dst = np.zeros(img_gray.shape)
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    return dst

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame,(3,3), 0)
    video = cv2.Canny(image=frame, threshold1=100, threshold2=200)
    if ret == True:
        #frame_out = cartoonize_image(frame, ds_factor=2, sketch_mode=True)
        frame_out = cartoonize_image(frame)
        cv2.imshow('Camara', frame_out)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllwindows()