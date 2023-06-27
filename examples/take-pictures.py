from aiymakerkit import vision
import time
import models

import numpy as np
# from cache import Cache
import os
import cv2
detector = vision.Detector(models.FACE_DETECTION_MODEL)
image_counter = 0
for frame in vision.get_frames():
    faces = detector.get_objects(frame, threshold=0.7)
    print(faces)
    for face in faces:

        bbox = face.bbox
        bounding_box = np.array([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]).astype("int")
        cropped_face = frame[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
        
        cv2.imwrite(f"person/face_{image_counter}.jpg", cropped_face)
        image_counter += 1
        time.sleep(0.5)# Save the image with bounding boxes

        # Display the cropped image
        # cv2.imshow('Cropped Face', cropped_face)
        # cv2.waitKey(0)  # waits until a key is pressed
        # cv2.destroyAllWindows()  # destroys the window showing image
        
        print(bounding_box)
    vision.draw_objects(frame, faces)
