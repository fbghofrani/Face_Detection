import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image_path = "image6.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error! Check the path")
    exit()
