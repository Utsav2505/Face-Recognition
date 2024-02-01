# Import necessary libraries
import numpy as np
import cv2 as cv

# Load the Haar Cascade for face detection
haar_cascade = cv.CascadeClassifier('E:\Documents\Python\Open CV\FaceDetection\HardCascades\haar_face.xml')

# Define a list of people for recognition
people = ['Udita', 'John Wick']

# Load the pre-trained features and labels
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

# Create an LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Read the pre-trained face recognizer model
face_recognizer.read('face_trained.yml')

# Read an image for testing
img = cv.imread(r'test2.JPG')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

# Iterate over each detected face and perform face recognition
for x, y, w, h in faces_rect:
    faces_roi = gray[y:y+w, x:x+h]

    # Predict the label and confidence for the detected face
    label, confidence = face_recognizer.predict(faces_roi)

    # Print the predicted label and confidence
    print(f'label = {people[label]} with a confidence of {confidence}')

    # Display the recognized label on the image
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness=1)

    # Draw a rectangle around the detected face
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

# Display the image with detected faces
cv.imshow('Detected Person', img)
cv.waitKey(0)
