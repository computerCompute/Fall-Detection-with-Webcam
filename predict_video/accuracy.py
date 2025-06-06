import sys
sys.path.append('/root/fall_detection/st-gcn')

from net.st_gcn import Model
import numpy as np
import torch
import torch.nn.functional as F
import pickle


# 데이터 및 라벨 로드
data = np.load('/root/fall_detection/dataset/saved_pkl/combined_mediapipe_30fps.npy')
print(f"{data.shape}")
with open('/root/fall_detection/dataset/saved_pkl/combined_mediapipe_30fps_labels.pkl', 'rb') as f:
    sample_names, class_labels = pickle.load(f)

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(in_channels=3, num_class=2, edge_importance_weighting=True,
              graph_args={'layout': 'openpose', 'strategy': 'uniform'})
model.load_state_dict(torch.load('/root/fall_detection/st-gcn/work_dir/shuffle_combined_model_f_mediapipe/epoch200_model.pt', map_location=device))
model.eval().to(device)

# 클래스 이름 정의
class_names = ['normal', 'falling']

# 예측 확인 루프
correct = 0
print(f"{'Idx':<4} {'Sample Name':<25} {'GT':<7} {'Pred':<7} {'✓?'}   Probabilities")
print("-" * 70)

for i in range(len(data)):
    sample = torch.tensor(data[i]).unsqueeze(0).float().to(device)  # (1, 3, 30, 18, 1)
    print(sample.shape)
    gt = class_labels[i]

    with torch.no_grad():
        output = model(sample)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)

    print(f"pred:{pred}")
    match = '✓' if pred == gt else '✗'
    print(f"{i:<4} {sample_names[i]:<25} {class_names[gt]:<7} {class_names[pred]:<7} {match}   {probs}")

    if pred == gt:
        correct += 1

# 전체 정확도 출력
print(f"\nAccuracy: {correct}/{len(data)} = {correct / len(data) * 100:.2f}%")

