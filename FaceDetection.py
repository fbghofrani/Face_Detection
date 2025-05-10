import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image_path = "image6.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error! Check the path")
    exit()

image = cv2.GaussianBlur(image, (5, 5), 0)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.equalizeHist(gray_image)


faces = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.05,
    minNeighbors=8,
    minSize=(40, 40)
)


output_folder = "faces"
os.makedirs(output_folder, exist_ok=True)

for i, (x, y, w, h) in enumerate(faces):
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (150, 150))
    face_path = os.path.join(output_folder, f"face_{i+1}.jpg")
    cv2.imwrite(face_path, face)
    cv2.imshow(f"Face {i+1}", face)

