import cv2
from scipy.spatial import distance as dist
import numpy as np
import dlib
import time
import os


def create_norm(mask):

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    norm = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if gray[i][j] == 255:
                norm[i][j] = np.ones(3)
            else:
                norm[i][j] = np.zeros(3)
    return norm


def shape_to_np(shape):

    coords = np.zeros((shape.num_parts, 2), dtype='int')

    for i in range(shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear


def weighted_smoothing(face, prev, smooth_factor):

    face = face * smooth_factor + prev * (1 - smooth_factor)
    return (face + 0.5).astype('int32')


haar_classifier = 'haarcascade_frontalface_alt.xml'
shape_pred = 'shape_predictor_68_face_landmarks.dat'

clf = cv2.CascadeClassifier(haar_classifier)
predictor = dlib.shape_predictor(shape_pred)

mask_directory = 'masks/'
masks = os.listdir(mask_directory)
num_masks = len(masks)
m = 0

COUNTER = 0
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 10

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

mask = cv2.imread(mask_directory + masks[m])
norm = create_norm(mask)

cam_port = 0

print("Iniciando la cámara...")

vs = cv2.VideoCapture(cam_port, cv2.CAP_DSHOW)
time.sleep(1.0)

trigger = True
prev = np.array([0, 0, 0, 0])
smooth_factor = .05

while True:

    ret, img = vs.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, 1.3, 5)
    for face in faces:

        if trigger:
            prev = face
            trigger = False
        else:
            face = weighted_smoothing(face, prev, smooth_factor)

        (x, y, w, h) = face

        rect = dlib.rectangle(x, y, (x + w),(y + w))
        shape = shape_to_np(predictor(gray, rect))

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        print(ear)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                m = (m + 1) % num_masks
                mask = cv2.imread(mask_directory + masks[m])
                norm = create_norm(mask)
                COUNTER = 0
        else:
            COUNTER = 0

        resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
        resized_norm = cv2.resize(norm, (w, h), interpolation=cv2.INTER_AREA)

        cropped_face = np.multiply(img[y:y + h, x:x + w], resized_norm) + resized_mask
        img[y:y + h, x:x + w] = cropped_face


        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

        prev = face



    cv2.imshow('Imagen con mascara', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


vs.release()
cv2.destroyAllWindows()
