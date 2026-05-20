import cv2
import numpy as np
from pose_extraction import PoseExtractor
from tensorflow.keras.models import load_model
from collections import deque

pose = PoseExtractor("C:/Users/saiva/Downloads/Rough/models/movenet.tflite")
model = load_model("C:/Users/saiva/Downloads/Rough/models/activity_model.h5")

actions = ['walking','jogging','running','handwaving']

sequence = []
predictions = deque(maxlen=5)

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    features = pose.extract_features(frame)

    sequence.append(features)

    if len(sequence) > 40:
        sequence.pop(0)

    if len(sequence) == 40:

        input_data = np.expand_dims(sequence, axis=0)

        prediction = model.predict(input_data)[0]

        action = actions[np.argmax(prediction)]

        predictions.append(action)

        # temporal smoothing
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

    cv2.imshow("HAR Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()