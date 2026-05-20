# Human Activity Recognition System

This project is a Human Activity Recognition System developed using TensorFlow, MoveNet, OpenCV, and LSTM-based Deep Learning techniques for real-time human activity detection and classification. The system uses pose estimation and skeletal tracking to recognize different human activities from webcam streams and video inputs.

The project extracts body keypoints using the MoveNet pose estimation model and generates skeletal representations of the human body. These keypoints are processed to calculate important motion features such as joint angles and body movement velocity. The extracted sequential features are then used to train an LSTM neural network for activity classification.

The system is capable of recognizing multiple activities including walking, jogging, running, and handwaving. It supports both real-time webcam detection and prerecorded video analysis. During prediction, the application displays live skeletal tracking along with the detected activity label on the video frame.

The project also includes dataset creation, frame extraction, model training, confusion matrix visualization, and classification report generation for performance evaluation. The trained model is saved and reused for future predictions.

## Features

* Real-time human activity recognition
* Pose estimation using MoveNet
* Skeleton and joint visualization
* LSTM-based sequential activity classification
* Webcam and video activity detection
* Automatic frame extraction and dataset generation
* Model training and evaluation
* Confusion matrix and classification report visualization

## Technologies Used

* Python
* TensorFlow
* OpenCV
* NumPy
* MoveNet
* LSTM Neural Networks
* Deep Learning
* Computer Vision

## Recognized Activities

* Walking
* Jogging
* Running
* Handwaving

## Applications

* Smart surveillance systems
* Fitness and sports analysis
* Human motion analysis
* Security monitoring
* Healthcare activity tracking
* AI-based behavior recognition

This project demonstrates the practical implementation of Artificial Intelligence, Deep Learning, Pose Estimation, and Computer Vision for real-time human activity analysis and recognition systems.
