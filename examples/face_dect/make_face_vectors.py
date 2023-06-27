import json
import numpy as np
import face_recognition
import os
import cv2
print(os.getcwd())
Person_da = False

from aiymakerkit import vision
import models
# Load known face encodings
known_face_encodings = []
known_face_names = []
for filename in os.listdir("person"):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        img_path = os.path.join("person", filename)
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            img_enc = encodings[0]
            known_face_encodings.append(img_enc)
            known_face_names.append(filename)

nested_lists = [arr.tolist() for arr in known_face_encodings]
print(nested_lists)
json_data = json.dumps(nested_lists)

with open("face_vectors.json", "w") as file:
    file.write(json_data)
