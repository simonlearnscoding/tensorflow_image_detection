import numpy as np
import face_recognition
import os
import cv2
print(os.getcwd())
Person_da = False

from aiymakerkit import vision
import models

personen = []
def create_person(name):
    if name in personen:
        return
    personen.append(Person(name))


class Person:
    def __init__(self, name):
        self.kommt = 0
        self.geht = 20
        self.ist_da = False
        self.name = name
        self.kommt_grenze = 10 
    

    def stop_action(self):
        print(f"bye {self.name}")

    def start_action(self):
        print(f"hi {self.name}")

# fabio = Person("fabio")



#Counter für kommen oder gehen    
    def check_if_there(self, gesehen):
        if self.name == gesehen:
            gesehen = True
        else:
            gesehen = False
        self.kommt_grenze = 10
        if self.ist_da:
            self.geht -= 1
            if gesehen:
                self.geht = 20
        else:
            if gesehen:
                self.kommt+=1

        if self.kommt > self.kommt_grenze:
            self.kommt = 0
            self.start_action(name)
            self.ist_da = True

        if self.geht == 0:
            self.geht = 20
            self.stop_action(name)
            self.ist_da = False
        

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

detector = vision.Detector(models.FACE_DETECTION_MODEL)

for frame in vision.get_frames():
    faces = detector.get_objects(frame, threshold=0.5)
    # print(faces)
    wurde_gesehen = False 
    for face in faces:

        bbox = face.bbox
        bounding_box = np.array([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]).astype("int")
        cropped_face = frame[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
        try:
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # print(e)
            pass

        #Get the face encodings for the current face
        face_encodings = face_recognition.face_encodings(cropped_face)
        # Compare the face with the known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
            if True in matches:


                matched_index = matches.index(True)
                matched_name = known_face_names[matched_index]
                name = matched_name.split("_")[0]
                
                """
                schau ob person schon gespeichert, wenn nicht, erstelle sie

                """
                
                create_person(name)
                wurde_gesehen_name = name
                
                for person in personen:
                    person.check_if_there(self, wurde_gesehen_name)

                cv2.putText(frame, name, (bounding_box[0], bounding_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    check_if_there(name, wurde_gesehen)
        # print(bounding_box)
    vision.draw_objects(frame, faces)

