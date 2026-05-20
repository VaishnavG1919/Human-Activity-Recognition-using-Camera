import tensorflow as tf
import numpy as np

class PoseExtractor:

    def __init__(self, model_path):

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.prev_keypoints = None

    def calculate_angle(self, a, b, c):

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def extract_features(self, frame):

        img = tf.image.resize_with_pad(
            tf.expand_dims(frame, axis=0), 192, 192
        )

        img = tf.cast(img, dtype=tf.uint8)
        img = img.numpy()

        self.interpreter.set_tensor(
            self.input_details[0]['index'], img
        )

        self.interpreter.invoke()

        keypoints = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0][0]

        keypoints_xy = keypoints[:, :2]

        # joint angles
        left_elbow = self.calculate_angle(keypoints_xy[5], keypoints_xy[7], keypoints_xy[9])
        right_elbow = self.calculate_angle(keypoints_xy[6], keypoints_xy[8], keypoints_xy[10])
        left_knee = self.calculate_angle(keypoints_xy[11], keypoints_xy[13], keypoints_xy[15])
        right_knee = self.calculate_angle(keypoints_xy[12], keypoints_xy[14], keypoints_xy[16])

        angles = [left_elbow, right_elbow, left_knee, right_knee]

        velocity = 0

        if self.prev_keypoints is not None:
            velocity = np.linalg.norm(keypoints_xy - self.prev_keypoints)

        self.prev_keypoints = keypoints_xy

        features = np.concatenate([
            keypoints.flatten(),
            angles,
            [velocity]
        ])

        return features

    # NEW FUNCTION (for skeleton drawing)
    def get_keypoints(self, frame):

        img = tf.image.resize_with_pad(
            tf.expand_dims(frame, axis=0), 192, 192
        )

        img = tf.cast(img, dtype=tf.uint8)
        img = img.numpy()

        self.interpreter.set_tensor(
            self.input_details[0]['index'], img
        )

        self.interpreter.invoke()

        keypoints = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0][0]

        return keypoints
    
print("Completed")