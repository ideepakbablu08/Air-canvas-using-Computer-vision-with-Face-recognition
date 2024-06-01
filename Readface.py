import cv2
import os
import csv

def store_data():
    name = input("Enter Name: ")
    while not name.isalpha():
        print("Please enter a valid name (alphabets only).")
        name = input("Enter Name: ")
    
    Id = input("Enter ID: ")
    while not Id.isdigit():
        print("Please enter a valid ID (digits only).")
        Id = input("Enter ID: ")

    return {'Name': name, 'ID': Id}

def TakeImages():
    dict1 = store_data()
    print(dict1)
    
    if dict1['ID'] == '1':
        fieldnames = ['Name', 'ID']
        with open('Profile.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(dict1)
    else:
        fieldnames = ['Name', 'ID']
        with open('Profile.csv', 'a+') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(dict1)

    cam = cv2.VideoCapture(0)

    # Haarcascade file for detection of face
    harcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    
    sampleNum = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            sampleNum += 1
            cv2.imwrite(f"TrainingImage/{dict1['Name']}.{dict1['ID']}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
        cv2.imshow('Capturing Face for Login', img)
        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 600:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Images saved for {dict1['Name']} with ID {dict1['ID']} in TrainingImage directory.")

TakeImages()
