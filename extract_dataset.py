import os
import cv2
import numpy as np
from pose_extraction import PoseExtractor

# Activity classes
actions = ['walking', 'jogging', 'running', 'handwaving']

# Load MoveNet pose extractor
pose = PoseExtractor("C:/Users/saiva/Downloads/Rough/models/movenet.tflite")

sequence_length = 40

X = []
y = []

frames_path = "frames"

for label, action in enumerate(actions):

    action_folder = os.path.join(frames_path, action)

    if not os.path.exists(action_folder):
        continue

    print(f"Processing {action}...")

    sequence = []

    for img_name in sorted(os.listdir(action_folder)):

        img_path = os.path.join(action_folder, img_name)

        frame = cv2.imread(img_path)

        if frame is None:
            continue

        features = pose.extract_features(frame)

        sequence.append(features)

        if len(sequence) == sequence_length:

            X.append(sequence)
            y.append(label)

            sequence = []

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)

np.save("C:/Users/saiva/Downloads/Rough/results/X_data.npy", X)
np.save("C:/Users/saiva/Downloads/Rough/results/y_data.npy", y)

print("Dataset extraction completed")