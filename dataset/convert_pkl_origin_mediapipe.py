import cv2
import numpy as np
import mediapipe as mp
import os
# 상수 정의
VIDEO_PATH = "path"
OUTPUT_PATH = "npy_path"

FRAME_RATE = 30
CHANNELS = 3  # x, y, confidence
NUM_KEYPOINTS = 18
NUM_PERSON = 1

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# COCO 순서에 맞춘 MediaPipe 인덱스 (neck은 None → 보간)
COCO_ORDERED_LANDMARKS = [
    0,      # nose
    None,   # neck (to be interpolated)
    11, 13, 15,  # left shoulder, elbow, wrist
    12, 14, 16,  # right shoulder, elbow, wrist
    23, 25, 27,  # left hip, knee, ankle
    24, 26, 28,  # right hip, knee, ankle
    5, 2,        # left eye, right eye
    6, 3         # left ear, right ear
]

# 비디오 로딩
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        print("video 인식 실패")
        break
    frames.append(frame)
print(f"총 프레임 수: {len(frames)}") 
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
        #skeleton  보간
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

# 저장
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.save(OUTPUT_PATH, data_numpy)
print("Saved to:", OUTPUT_PATH)
print("Shape:", data_numpy.shape)  # (N, 3, 30, 18, 1)
