# Face Recognition Project

## Overview
This project implements a simple face recognition system using OpenCV in Python. The system uses Haar Cascade for face detection and LBPH (Local Binary Pattern Histograms) for face recognition. It is designed to recognize faces of specific individuals in images.

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy

## Project Structure
- `Train/`: Directory containing training images for recognized individuals.
- `face_detection/`: Directory containing the Haar Cascade XML file for face detection.
- `features.npy`: NumPy file storing extracted features from training images.
- `labels.npy`: NumPy file storing corresponding labels for the training data.
- `face_trained.yml`: Trained LBPH face recognizer model.
- `test_image.jpg`: Sample image for testing the face recognition system.
- `main_train.py`: Script for training the face recognition model.
- `main_recognize.py`: Script for testing the face recognition on a new image.

## How it Works
1. **Data Collection and Preparation:**
   - Training images of recognized individuals are stored in the `Train/` directory.
   - The `main_train.py` script is used to read and preprocess these images, extracting features and labels.
   - The features and labels are saved as NumPy arrays (`features.npy` and `labels.npy`).

2. **Training the Face Recognizer:**
   - The LBPH face recognizer is created using OpenCV.
   - It is trained on the extracted features and corresponding labels using the `train()` method.
   - The trained model is saved as `face_trained.yml`.

3. **Testing Face Recognition:**
   - The `main_recognize.py` script loads the pre-trained model and the test image.
   - Haar Cascade is used to detect faces in the test image.
   - For each detected face, the LBPH face recognizer predicts the label and confidence.
   - The recognized label and confidence are printed, and the result is displayed on the image.
