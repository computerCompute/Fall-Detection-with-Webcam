import numpy as np
import pickle
from sklearn.utils import shuffle

input_data_path = '/root/fall_detection/dataset/saved_pkl/combined_mediapipe_30fps.npy'
input_label_path = '/root/fall_detection/dataset/saved_pkl/combined_mediapipe_30fps_labels.pkl'


data = np.load(input_data_path)
with open(input_label_path, 'rb') as f:
    labels = pickle.load(f)  # labels = [filenames, label_list]

filenames = labels[0]
label_list = labels[1]

combined = list(zip(data, filenames, label_list))
combined = shuffle(combined, random_state=42)

data = np.array([x[0] for x in combined])
filenames = [x[1] for x in combined]
label_list = [x[2] for x in combined]

output_data_path = '/root/fall_detection/dataset/saved_pkl/shuffled_combined_mediapipe_30fps.npy'
output_label_path = '/root/fall_detection/dataset/saved_pkl/shuffled_combined_mediapipe_30fps_labels.pkl'

output_data_path_2 = '/root/fall_detection/dataset/saved_pkl/shuffled_combined_mediapipe_30fps.npy'
output_label_path_2 = '/root/fall_detection/dataset/saved_pkl/shuffled_combined_mediapipe_30fps_labels.pkl'


np.save(output_data_path, data)
with open(output_label_path, 'wb') as f:
    pickle.dump([filenames, label_list], f)

np.save(output_data_path_2, data)
with open(output_label_path_2, 'wb') as f:
    pickle.dump([filenames, label_list], f)

print('\n', data.shape)
print('\n', filenames)
print('\n', label_list)
print(output_data_path)
print(output_data_path_2)
