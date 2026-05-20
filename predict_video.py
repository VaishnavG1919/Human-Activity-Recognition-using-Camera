import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("C:/Users/saiva/Downloads/HAR/results/activity_model.h5")

activities = ["walking", "running", "jogging", "handwaving", "handclapping", "boxing"]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

video_path = "C:/Users/saiva/Downloads/HAR/inference/test_video.mp4"

cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = pose.process(img_rgb)

    if result.pose_landmarks:

        keypoints = []

        for lm in result.pose_landmarks.landmark:
            keypoints.append(lm.x)
            keypoints.append(lm.y)

        keypoints = np.array(keypoints).reshape(1,1,-1)

        prediction = model.predict(keypoints)

        label = activities[np.argmax(prediction)]

        cv2.putText(frame, label, (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()