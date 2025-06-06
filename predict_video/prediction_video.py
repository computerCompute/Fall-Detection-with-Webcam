import sys
sys.path.append('/fall_detection/st-gcn')
from net.st_gcn import Model
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import subprocess


class PoseDataset(Dataset):
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx]).float().to(self.device)
        return sample

def process_video_with_text(input_video_path, output_video_path, dataloader, model, device):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Failed to open video file {input_video_path}")
        return

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input video FPS: {input_fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

    model.to(device)
    model.eval()

    time = 1
    for samples in dataloader:
        samples = samples.to(device)
#        print(samples.size()) [1,3,30,18,1]

        with torch.no_grad():
            outputs = model(samples) #[1,2]
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        for batch_idx in range(samples.size(0)):
            label = 'Normal' if preds[batch_idx] == 0 else 'Fall'
            print(f"{time} :  {label}")

            text_color = (0, 255, 0) if label == 'Normal' else (0, 0, 255)

            for i in range(30):
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame.")
                    break

                cv2.putText(frame, f'Prediction: {label}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                out.write(frame)

                time += 1

    cap.release()
    out.release()


data = np.load('npy_path')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#st-gcn
model = Model(in_channels=3, num_class=2, edge_importance_weighting=True,
              graph_args={'layout': 'openpose', 'strategy': 'uniform'})
model.load_state_dict(torch.load('model_path', map_location=device))
model.eval().to(device)

dataset = PoseDataset(data, device)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#input_video_path = 'falling_original_30fps.mp4'
OPENPOSE_BIN = '/fall_detection/openpose/bin/openpose.bin'

cmd = [
    OPENPOSE_BIN,
    '--video', '/root/fall_detection/data/falling_video/falling_original_30fps.mp4',
    '--write_video', '/root/fall_detection/data/falling_video/falling_original_30fps_mediapipe.mp4',
    '--model_pose', 'COCO',
    '--model_folder', '/root/install_openpose/openpose/models',
    '--display', '0',
    '--render_pose', '1',
    '--number_people_max', '1'
]
subprocess.run(cmd)


input_video_path = 'video_path'
output_video_path = 'video_path'

process_video_with_text(input_video_path, output_video_path, dataloader, model, device)
