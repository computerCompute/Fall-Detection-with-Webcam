import os
import cv2
import numpy as np
import pickle
import mediapipe as mp
from tqdm import tqdm

VIDEO_DIRECTORIES = [
    "/root/fall_detection/data/falling_video/falling_30fps",
    "/root/fall_detection/data/normal_video/f_normal_30fps"
]

OUTPUT_BASE_PATH = "/root/fall_detection/dataset/saved_pkl"
FRAME_RATE = 30
CHANNELS = 3
NUM_KEYPOINTS = 18
NUM_PERSON = 1

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

COCO_ORDERED_LANDMARKS = [
    0, None, 11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28, 5, 2, 6, 3
]

def process_video(video_path, label_value):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    T = len(frames)
    num_segments = T // FRAME_RATE
    data_numpy = np.zeros((num_segments, CHANNELS, FRAME_RATE, NUM_KEYPOINTS, NUM_PERSON))

    for t, frame in enumerate(frames):
        segment_idx = t // FRAME_RATE
        frame_idx = t % FRAME_RATE

        if segment_idx >= num_segments:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            continue

        img_h, img_w = frame.shape[:2]

        for k, idx in enumerate(COCO_ORDERED_LANDMARKS):
            if idx is None:
                l = results.pose_landmarks.landmark[11]
                r = results.pose_landmarks.landmark[12]
                x = (l.x + r.x) / 2
                y = (l.y + r.y) / 2
                conf = (l.visibility + r.visibility) / 2
            else:
                lm = results.pose_landmarks.landmark[idx]
                x, y, conf = lm.x, lm.y, lm.visibility

            x_px = x * img_w
            y_px = y * img_h

            data_numpy[segment_idx, 0, frame_idx, k, 0] = x_px
            data_numpy[segment_idx, 1, frame_idx, k, 0] = y_px
            data_numpy[segment_idx, 2, frame_idx, k, 0] = conf

    label_list = [label_value for _ in range(num_segments)]
    return data_numpy, label_list


def process_dir(input_dir, label_value):
    all_data = []
    all_labels = []
    sample_names = []

    for video_file in tqdm(sorted(os.listdir(input_dir)), desc=f"Processing {input_dir}"):
        video_path = os.path.join(input_dir, video_file)
        if video_path.endswith('.mp4'):
            print(f"Processing file: {video_file}")  # 처리 중인 파일명 출력
            data_numpy, label_list = process_video(video_path, label_value)
            num_segments = data_numpy.shape[0]

            for segment_idx in range(num_segments):
                sample_names.append(f"{video_file}_{segment_idx}")
                all_data.append(data_numpy[segment_idx])
                all_labels.append(label_list[segment_idx])

    return sample_names, all_data, all_labels


abnormal_dir = "/root/fall_detection/data/falling_video/falling_30fps"
normal_dir = "/root/fall_detection/data/normal_video/f_normal_30fps"

ab_names, ab_data, ab_labels = process_dir(abnormal_dir, 1)
nm_names, nm_data, nm_labels = process_dir(normal_dir, 0)

sample_names = ab_names + nm_names
data_list = ab_data + nm_data
label_list = ab_labels + nm_labels

data_numpy = np.stack(data_list)

output_data_path = os.path.join(OUTPUT_BASE_PATH, "combined_mediapipe_30fps.npy")
output_label_path = os.path.join(OUTPUT_BASE_PATH, "combined_mediapipe_30fps_labels.pkl")

os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
np.save(output_data_path, data_numpy)

with open(output_label_path, 'wb') as f:
    pickle.dump((sample_names, label_list), f)

print("   - data_30fps (.npy):", output_data_path)
print("   - label_30fps (.pkl):", output_label_path)
print("Shape of data:", data_numpy.shape)
print("Number of labels:", len(label_list))
