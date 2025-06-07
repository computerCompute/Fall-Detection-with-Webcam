from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from net.st_gcn import Model

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(in_channels=3, num_class=2, edge_importance_weighting=True,
              graph_args={'layout': 'openpose', 'strategy': 'uniform'})
model.load_state_dict(torch.load('epoch200_model.pt', map_location=device))
model.eval().to(device)

class_names = ['Normal', 'Fall']

COCO_ORDERED_LANDMARKS = [
    0, None, 11, 13, 15, 12, 14, 16,
    23, 25, 27, 24, 26, 28,
    5, 2, 6, 3
]

frame_buffer = []
FRAME_LEN = 30
NUM_JOINTS = 18
CHANNELS = 3
MAX_PERSON = 1

def extract_keypoints(results, img_h, img_w):
    keypoints = np.zeros((CHANNELS, NUM_JOINTS, MAX_PERSON))
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        for k, idx in enumerate(COCO_ORDERED_LANDMARKS):
            if idx is None:
                l = lm[11]
                r = lm[12]
                x = (l.x + r.x) / 2
                y = (l.y + r.y) / 2
                conf = (l.visibility + r.visibility) / 2
            else:
                point = lm[idx]
                x, y, conf = point.x, point.y, point.visibility
            keypoints[0, k, 0] = x * img_w
            keypoints[1, k, 0] = y * img_h
            keypoints[2, k, 0] = conf
    return keypoints

def predict_if_ready():
    if len(frame_buffer) == FRAME_LEN:
        np_data = np.stack(frame_buffer, axis=1)
        input_tensor = torch.tensor(np_data, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)
        frame_buffer.pop(0)
        return class_names[pred], probs
    return "Loading...", [0.0, 0.0]

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        keypoints = extract_keypoints(results, h, w)
        frame_buffer.append(keypoints)
        label, probs = predict_if_ready()
        color = (0, 255, 0) if label == 'Normal' else (0, 0, 255)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f'Prediction: {label}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=####)
