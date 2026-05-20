import cv2
import os

video_folder = "C:/Users/saiva/Downloads/Rough/dataset"
output_folder = "C:/Users/saiva/Downloads/Rough/frames"

os.makedirs(output_folder, exist_ok=True)

FRAME_SKIP = 15   # Save 1 frame every 15 frames

for activity in os.listdir(video_folder):
    activity_path = os.path.join(video_folder, activity)

    if not os.path.isdir(activity_path):
        continue

    for video in os.listdir(activity_path):

        video_path = os.path.join(activity_path, video)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        saved_count = 0

        save_folder = os.path.join(output_folder, activity)
        os.makedirs(save_folder, exist_ok=True)

        print(f"\nProcessing video: {video}")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Save only every 15th frame
            if frame_count % FRAME_SKIP == 0:
                frame_name = f"{video}_frame_{saved_count}.jpg"
                cv2.imwrite(os.path.join(save_folder, frame_name), frame)

                saved_count += 1
                print(f"Saved frame {saved_count}")

            frame_count += 1

        cap.release()

print("\nFrames extracted successfully")