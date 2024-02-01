import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('E:\Documents\Python\Open CV\FaceDetection\HardCascades\haar_face.xml')

people = ['Udita','John Wick']
# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'Train\Udita\photo_2024-01-31_19-51-29 (2).jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Person',gray)

# detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
for x,y,w,h in faces_rect:
    faces_roi = gray[y:y+w,x:x+h]

    label,confidence = face_recognizer.predict(faces_roi)
    print(f'label = {people[label]} with a confidence of {confidence}')

    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_PLAIN,1.0,(0,255,0),thickness=1)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Detected Person',img)
cv.waitKey(0)