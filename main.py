import os
import cv2
import time
import numpy as np
import tensorflow as tf
from collections import deque
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

DATASET_PATH = "C:/Users/saiva/Downloads/Pose Detection/dataset"
FRAMES_PATH = "C:/Users/saiva/Downloads/Pose Detection/frames"
RESULTS_PATH = "C:/Users/saiva/Downloads/Pose Detection/results"
MODEL_PATH = "C:/Users/saiva/Downloads/Pose Detection/models/activity_model.keras"
MOVENET_MODEL = "C:/Users/saiva/Downloads/Pose Detection/models/movenet.tflite"

actions = ['walking','jogging','running','handwaving']

FRAME_SKIP = 15
SEQUENCE_LENGTH = 40

os.makedirs(FRAMES_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs("models", exist_ok=True)

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
    for y, x, c in keypoints:
        if c > threshold:
            cv2.circle(frame,(int(x*w),int(y*h)),5,(0,255,0),-1)

    # draw bones
    for p1, p2 in EDGES:

        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]

        if c1 > threshold and c2 > threshold:
            cv2.line(
                frame,
                (int(x1*w), int(y1*h)),
                (int(x2*w), int(y2*h)),
                (0,255,255),
                2
            )

class PoseExtractor:

    def __init__(self, model_path):

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.prev_keypoints = None

    def calculate_angle(self,a,b,c):

        a=np.array(a)
        b=np.array(b)
        c=np.array(c)

        ba=a-b
        bc=c-b

        cosine_angle=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
        angle=np.arccos(cosine_angle)

        return np.degrees(angle)

    def extract_features(self,frame):

        img=tf.image.resize_with_pad(
            tf.expand_dims(frame,axis=0),192,192
        )

        img=tf.cast(img,dtype=tf.uint8).numpy()

        self.interpreter.set_tensor(self.input_details[0]['index'],img)
        self.interpreter.invoke()

        keypoints=self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0][0]

        keypoints_xy=keypoints[:,:2]

        left_elbow=self.calculate_angle(keypoints_xy[5],keypoints_xy[7],keypoints_xy[9])
        right_elbow=self.calculate_angle(keypoints_xy[6],keypoints_xy[8],keypoints_xy[10])
        left_knee=self.calculate_angle(keypoints_xy[11],keypoints_xy[13],keypoints_xy[15])
        right_knee=self.calculate_angle(keypoints_xy[12],keypoints_xy[14],keypoints_xy[16])

        angles=[left_elbow,right_elbow,left_knee,right_knee]

        velocity=0

        if self.prev_keypoints is not None:
            velocity=np.linalg.norm(keypoints_xy-self.prev_keypoints)

        self.prev_keypoints=keypoints_xy

        features=np.concatenate([
            keypoints.flatten(),
            angles,
            [velocity]
        ])

        return features

    def get_keypoints(self,frame):

        img=tf.image.resize_with_pad(
            tf.expand_dims(frame,axis=0),192,192
        )

        img=tf.cast(img,dtype=tf.uint8).numpy()

        self.interpreter.set_tensor(self.input_details[0]['index'],img)
        self.interpreter.invoke()

        keypoints=self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0][0]

        return keypoints

def extract_frames():

    for activity in os.listdir(DATASET_PATH):

        activity_path=os.path.join(DATASET_PATH,activity)

        if not os.path.isdir(activity_path):
            continue

        for video in os.listdir(activity_path):

            video_path=os.path.join(activity_path,video)
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print("Error: Cannot open video file.")
                return

            frame_count=0
            saved_count=0

            save_folder=os.path.join(FRAMES_PATH,activity)
            os.makedirs(save_folder,exist_ok=True)

            print("Processing:",video)

            while True:

                ret,frame=cap.read()

                if not ret:
                    break

                if frame_count%FRAME_SKIP==0:

                    name=f"{video}_{saved_count}.jpg"
                    cv2.imwrite(os.path.join(save_folder,name),frame)

                    saved_count+=1

                frame_count+=1

            cap.release()

    print("Frames extracted")

def create_dataset():

    pose=PoseExtractor(MOVENET_MODEL)

    X=[]
    y=[]

    for label,action in enumerate(actions):

        folder=os.path.join(FRAMES_PATH,action)

        if not os.path.exists(folder):
            continue

        sequence=[]

        for img in sorted(os.listdir(folder)):

            path=os.path.join(folder,img)

            frame=cv2.imread(path)

            if frame is None:
                continue

            features=pose.extract_features(frame)

            sequence.append(features)

            if len(sequence)==SEQUENCE_LENGTH:

                X.append(sequence)
                y.append(label)

                sequence=[]

    X=np.array(X)
    y=np.array(y)

    np.save(os.path.join(RESULTS_PATH,"X_data.npy"),X)
    np.save(os.path.join(RESULTS_PATH,"y_data.npy"),y)

    print("Dataset created:",X.shape)

def train_model():

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report

    X = np.load(os.path.join(RESULTS_PATH,"X_data.npy"))
    y = np.load(os.path.join(RESULTS_PATH,"y_data.npy"))

    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Sequential()

    model.add(LSTM(64, return_sequences=True,
                   input_shape=(SEQUENCE_LENGTH, X.shape[2])))
    model.add(Dropout(0.3))

    model.add(LSTM(32))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=16,
        validation_data=(X_test, y_test)
    )

    model.save(MODEL_PATH)

    print("Model trained and saved")

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=actions,
                yticklabels=actions)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # -----------------------------
    # CLASSIFICATION REPORT
    # -----------------------------
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred_classes,
                                target_names=actions))

def run_webcam():

    pose = PoseExtractor(MOVENET_MODEL)
    model = load_model(MODEL_PATH)

    sequence = []
    predictions = deque(maxlen=5)

    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    timestamp = int(time.time())
    video_path = os.path.join(RESULTS_PATH, f"webcam_output_{timestamp}.avi")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))

    print("Recording started...")

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break
        
        keypoints = pose.get_keypoints(frame)
        draw_skeleton(frame, keypoints)

        features = pose.extract_features(frame)

        sequence.append(features)

        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        if len(sequence) == SEQUENCE_LENGTH:

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

    print("Video saved in results folder:", video_path)

def run_video_detection(video_path):

    pose = PoseExtractor(MOVENET_MODEL)
    model = load_model(MODEL_PATH)

    sequence = []
    predictions = deque(maxlen=5)

    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    output_video = os.path.join(RESULTS_PATH, "video_prediction.avi")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 20.0,
                          (frame_width, frame_height))

    print("Processing video...")

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        keypoints = pose.get_keypoints(frame)
        draw_skeleton(frame, keypoints)

        features = pose.extract_features(frame)

        sequence.append(features)

        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        if len(sequence) == SEQUENCE_LENGTH:

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

        cv2.imshow("Video Activity Detection", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Output video saved at:", output_video)

while True:

    print("\nHuman Activity Recognition Pipeline")

    print("1 - Extract Frames")
    print("2 - Create Dataset")
    print("3 - Train Model")
    print("4 - Run Webcam Detection")
    print("5 - Detect Activity from Video")
    print("6 - Exit Program")

    choice = input("Select step: ")

    if choice == "1":
        extract_frames()

    elif choice == "2":
        create_dataset()

    elif choice == "3":
        train_model()

    elif choice == "4":
        run_webcam()

    elif choice == "5":
        video_path = input("Enter video path: ")
        run_video_detection(video_path)

    elif choice == "6":
        print("Program stopped.")
        break

    else:
        print("Invalid choice. Please try again.")