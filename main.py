import numpy as np
import face_recognition
import os
import cv2
from aiymakerkit import vision
import models

personen = {}

def create_person(name):
    print(f'erkanntes gesicht {name}')
    if name in personen:
        return
    personen[name] = Person(name)
    print("created person")
    print (personen)


class Person:
    def __init__(self, name):
        self.kommt = 0
        self.geht = 0
        self.ist_da = False
        self.name = name
        self.kommt_grenze = 5 
        self.geht_grenze = 5 
    

    def stop_action(self):
        print(f"bye {self.name}")

    def start_action(self):
        print(f"hi {self.name}")

# fabio = Person("fabio")



#Counter für kommen oder gehen    
    def check_if_there(self, personen_die_gesehen_wurden):
        # print(f"person name: {self.name}, gesehen name {gesehen}")
        
        if self.name in personen_die_gesehen_wurden:
            if self.ist_da: # GESEHEN UND DA
                # print(f"{self.name} wurde gesehen. geht: {self.geht} ")
                self.geht = 0 # resette geht timer
            else: # GESEHEN UND NICHT DA
                # print(f"{self.name} kommt: {self.kommt}")
                self.kommt += 1
                # print(f"kommt = {self.kommt}")
                if self.kommt >= self.kommt_grenze:
                    self.start_action()
                    self.kommt = 0
                    self.ist_da = True
        else:
            if self.ist_da: # NICHT GESEHEN UND DA
                # print(f"{self.name} wurde nicht gesehen geht: {self.geht}")
                self.geht += 1
                if self.geht > self.geht_grenze:
                    self.stop_action()
                    self.ist_da = False
                    self.geht = 0
            else: # NICHT GESEHEN UND NICHT DA
                self.kommt = 0
                # print(f"{self.name} kommt: {self.kommt}")
        

        # print(f"wurde {self.name} gesehen?? {gesehen}")
           

        

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

def check_alle_personen(personen_die_gesehen_wurden):
    for person in personen:
        personen[person].check_if_there(personen_die_gesehen_wurden)

for frame in vision.get_frames():
    faces = detector.get_objects(frame, threshold=0.5)
    personen_die_gesehen_wurden = [] 
    for face in faces:
        bbox = face.bbox
        bounding_box = np.array([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]).astype("int")
        cropped_face = frame[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
        try:
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        except Exception as e:
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
                personen_die_gesehen_wurden.append(name)
                cv2.putText(frame, name, (bounding_box[0], bounding_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    
    check_alle_personen(personen_die_gesehen_wurden)
    
    vision.draw_objects(frame, faces)

