import cv2
import os

START_TIME_MS = 2000
END_TIME_MS = 4000

def extract_frames(video_path, out_dir, every_n=2):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME_MS)
    i, saved = 0, 0
    while cap.isOpened():
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
    
        if current_time > END_TIME_MS:
            break

        ret, frame = cap.read()
        if not ret:
            break
        if i % every_n == 0:
            cv2.imwrite(f"{out_dir}_frame_{saved:04d}.jpg", frame)
            saved += 1
        i += 1
    cap.release()

directory = './data/videos'

for filename in os.listdir(directory):
    # Construct full path to verify it is a file, not a folder
    full_path = os.path.join(directory, filename)
    if os.path.isfile(full_path):
        partial_filname = filename.split('.')[0]
        extract_frames('./data/videos/{}.mp4'.format(partial_filname), './data/frames/{}'.format(partial_filname), 3)