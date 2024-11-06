import numpy as np
import glob
import re
import OpenRadar.mmwave.dsp as dsp
from OpenRadar.mmwave.dataloader import DCA1000
from helper import *
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Define the path to your data files
data_path = 'drive/MyDrive/Data/*.npy'
mmwave_files = glob.glob(data_path)

# Lists to store processed data, labels, and subjects for all radar cubes
features_1 = []
features_2 = []
features_3 = []
labels = []
subjects = []  # New list to store subject names

# Define class names based on your project
class_names = ['amusing', 'relaxed', 'boring', 'scary']

# Regular expression to parse the filename format
file_pattern = re.compile(r'([^_/]+)_([^_/]+)_\d+-\d+-\d+-\d+-\d+-\d+\.npy')

for file in mmwave_files:
    # Extract class and subject names from filename
    match = file_pattern.search(file)
    if not match:
        print(f"Filename format not recognized for file: {file}")
        continue

    class_name, subject_name = match.groups()

    # Only proceed if class_name is valid
    if class_name not in class_names:
        print(f"Class name {class_name} not recognized, skipping file.")
        continue

    # Load and preprocess the data
    with open(file, 'rb') as f:
        data = np.load(f)

        # Process each of the 600 radar cubes in the data
        for radar_cube in data:
            # Organize, process range, and subtract mean
            radar_cube = np.apply_along_axis(DCA1000.organize, 0, radar_cube, num_chirps=182*3, num_rx=4, num_samples=256)
            radar_cube = dsp.range_processing(radar_cube)
            mean = radar_cube.mean(0)
            radar_cube = radar_cube - mean
            radar_cube = np.concatenate((radar_cube[0::3, ...], radar_cube[1::3, ...], radar_cube[2::3, ...]), axis=1)
            radar_cube = radar_cube.reshape(182, 3, 4, 256)
            doppler_fft = dopplerFFT(radar_cube)
            pcds = frame2pointcloud(doppler_fft)

            # Extract Feature 1 (shape: 256)
            rangeOutput = np.transpose(np.absolute(radar_cube), (1, 2, 0, 3)).sum(axis=(0, 1, 2))
            features_1.append(rangeOutput)

            # Extract Feature 2 (shape: 182, 256)
            rangeDoppler = np.absolute(doppler_fft).sum(axis=(0, 1))
            features_2.append(rangeDoppler)

            # Extract Feature 3 (assuming pcds has shape (1600, 6) as described)
            features_3.append(pcds)

            # Add the class label and subject name for each radar cube
            labels.append(class_name)
            subjects.append(subject_name)  # Store the subject name

# Convert lists to numpy arrays
features_1 = np.array(features_1)  # shape: (num_cubes_total, 256)
features_2 = np.array(features_2)  # shape: (num_cubes_total, 182, 256)
features_3 = np.array(features_3)  # shape: (num_cubes_total, 1600, 6)
labels = np.array(labels)
subjects = np.array(subjects)  # Convert subjects to array

# Encode labels as integers and one-hot encode
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded, num_classes=len(class_names))

# Save features, labels, and subjects to .npz file

# Print the shapes to verify
print("Feature 1 shape:", features_1.shape)
print("Feature 2 shape:", features_2.shape)
print("Feature 3 shape:", features_3.shape)
print("Labels shape:", labels_categorical.shape)
print("Subjects shape:", subjects.shape)
