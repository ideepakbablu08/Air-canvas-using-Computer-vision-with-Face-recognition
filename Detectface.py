import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd

# Function to detect the face
def DetectFace():
    # Load profile data
    reader = csv.DictReader(open('Profile.csv'))
    print('Detecting Login Face')
    for row in reader:
        result = dict(row)
        if result['ID'] == '1':
            name1 = result['Name']
        elif result['ID'] == '2':
            name2 = result['Name']

    # Load face recognition model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainData/Trainner.yml")

    # Load cascade classifier for face detection
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
    
    # Initialize camera
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    Face_Id = ''

    # Camera loop
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        # Face detection loop
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            Face_Id = 'Not detected'
            Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            
            # Recognized face
            if confidence < 80:
                if Id == 1:
                    Predicted_name = name1
                elif Id == 2:
                    Predicted_name = name2
                Face_Id = Predicted_name
            else:
                Predicted_name = 'Unknown'
                Face_Id = Predicted_name
                
            cv2.putText(frame, str(Predicted_name), (x, y + h), font, 1, (255, 255, 255), 2)

        cv2.imshow('Picture', frame)
        cv2.waitKey(1)

        # Checking if the face matches for Login
        if Face_Id == 'Not detected':
            print("Face Not Detected. Try again.")
            pass
        elif Face_Id == name1 or name2 and Face_Id != 'Unknown':
            print('Detected as {}. Login successful.'.format(name1))
            print('Welcome, {}.'.format(name1))
            break
        else:
            print('Login failed. Please try again.')

    cam.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    DetectFace()
