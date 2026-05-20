import cv2
import numpy as np
import os
import time
from pose_extraction import PoseExtractor
from tensorflow.keras.models import load_model
from collections import deque

# Create results folder
os.makedirs("results", exist_ok=True)

# Skeleton edges
EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,7),(7,9),
    (6,8),(8,10),(5,6),(5,11),
    (6,12),(11,12),(11,13),(13,15),
    (12,14),(14,16)
]

def draw_skeleton(frame, keypoints, threshold=0.3):

    h, w, _ = frame.shape
    keypoints = keypoints.reshape(-1,3)

    # draw joints
    for y,x,c in keypoints:
        if c > threshold:
            cv2.circle(frame,(int(x*w),int(y*h)),5,(0,255,0),-1)

    # draw bones
    for p1,p2 in EDGES:

        y1,x1,c1 = keypoints[p1]
        y2,x2,c2 = keypoints[p2]

        if c > threshold and c2 > threshold:
            cv2.line(
                frame,
                (int(x1*w),int(y1*h)),
                (int(x2*w),int(y2*h)),
                (0,255,255),
                2
            )

# Load models
pose = PoseExtractor("C:/Users/saiva/Downloads/Pose Detection/models/movenet.tflite")
model = load_model("C:/Users/saiva/Downloads/Pose Detection/models/activity_model.h5")

actions = ['walking','jogging','running','handwaving']

sequence = []
predictions = deque(maxlen=5)

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Save video
timestamp = int(time.time())
video_path = f"C:/Users/saiva/Downloads/Pose Detection/results/output_{timestamp}.avi"

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))

print("Recording started...")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    features = pose.extract_features(frame)
    keypoints = pose.get_keypoints(frame)

    draw_skeleton(frame, keypoints)

    sequence.append(features)

    if len(sequence) > 40:
        sequence.pop(0)

    if len(sequence) == 40:

        input_data = np.expand_dims(sequence, axis=0)

        prediction = model.predict(input_data, verbose=0)[0]

        action = actions[np.argmax(prediction)]

        predictions.append(action)

        final_action = max(set(predictions), key=predictions.count)

        cv2.putText(
            frame,
            final_action,
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    out.write(frame)

    cv2.imshow("HAR Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved in results folder.")