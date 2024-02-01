# Import necessary libraries
import os
import cv2 as cv
import numpy as np

# Define a list of people to recognize
people = ['John Wick']

# Specify the directory where the training data is located
DIR = r'Train'

# Load the Haar Cascade for face detection
haar_cascade = cv.CascadeClassifier('Haar_Cascades\haar_face.xml')

# Initialize empty lists to store features and labels for training
features = []
labels = []

# Function to create training data
def create_train():
    for person in people:
        # Construct the path for each person's images
        path = os.path.join(DIR, person)
        # Assign a unique label to each person
        label = people.index(person)

        # Loop through each image in the person's directory
        for img in os.listdir(path):
            # Construct the full path for the image
            img_path = os.path.join(path, img)

            # Read the image and convert it to grayscale
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

            # Iterate over each detected face and extract the region of interest
            for x, y, w, h in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                # Append the face region to the features list and the corresponding label to the labels list
                features.append(faces_roi)
                labels.append(label)

# Call the function to create the training data
create_train()
print('Training Done---------')

# Convert features list to NumPy array with 'object' datatype
features = np.array(features, dtype='object')
labels = np.array(labels)

# Create an LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer using the features and labels
face_recognizer.train(features, labels)

# Save the trained face recognizer model to a file
face_recognizer.save('face_trained.yml')

# Save the features and labels to NumPy arrays
np.save('features.npy', features)
np.save('labels.npy', labels)
