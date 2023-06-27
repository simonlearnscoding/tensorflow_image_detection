from aiymakerkit import vision
from time import sleep
import models
name = input("Inserire il nome del user: ")
amount_pics = int(input("Quante Foto facciamo? "))

import numpy as np
# from cache import Cache
import os
import cv2


def save_image(face, frame):
    bbox = face.bbox
    bounding_box = np.array([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]).astype("int")
    cropped_face = frame[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
    return cropped_face

def take_pictures(amount):
    print(f"Ciao {name}! Mostrate il proprio viso davanti alla telecamera e muovete la testa in diverse posizioni.")
    sleep(3)
    detector = vision.Detector(models.FACE_DETECTION_MODEL)
    image_counter = 0
    for frame in vision.get_frames():
        faces = detector.get_objects(frame, threshold=0.7)
        
        if len(faces) == 0:
            continue
        
        if len(faces) > 1:
            print("Solo una faccia per piacere")
            continue
        
        face = faces[0]
        cropped_face = save_image(face, frame)
        cv2.imwrite(f"person/{name}_{image_counter}.jpg", cropped_face)
        print(f"{image_counter} foto fatte")
        image_counter += 1
        sleep(1)# Save the image with bounding boxes
            
        if image_counter >= amount:
            print(f"Grazie {name}!!")
            break
        vision.draw_objects(frame, faces)

    
take_pictures(amount_pics)
