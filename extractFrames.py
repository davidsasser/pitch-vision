import cv2
import os

def extract_frames(video_path, out_dir, every_n=2):
    cap = cv2.VideoCapture(video_path)
    i, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % every_n == 0:
            cv2.imwrite(f"{out_dir}/frame_{saved:04d}.jpg", frame)
            saved += 1
        i += 1
    cap.release()

directory = './data/videos'

for filename in os.listdir(directory):
    # Construct full path to verify it is a file, not a folder
    full_path = os.path.join(directory, filename)
    if os.path.isfile(full_path):
        print(filename)
        break