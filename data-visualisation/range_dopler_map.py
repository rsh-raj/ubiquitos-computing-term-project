import numpy as np
data = np.load('drive/MyDrive/processed_data_sub.npz')
rangeDoppler_data = data['features_2']  # Assuming 'features_2' is the range-doppler map with shape (num_samples, 182, 256)
labels = data['labels']  # Assuming labels correspond to classes like 'amusing', 'relaxed', etc.
subjects = data['subjects']  # Assuming 'subjects' contains the subject names like 'astitva', 'pranil', 'rishi'
import numpy as np
import matplotlib.pyplot as plt

# Load processed data
# data = np.load('processed_data.npz')
# rangeDoppler_data = data['features_2']
# labels = data['labels']
# subjects = data['subjects']

# Define class names and subjects
class_names = ['amusing', 'relaxed', 'boring', 'scary']
subject_names = ['astitva', 'pranil', 'rishi']
num_samples_to_plot = 1  # One sample per class and subject

# Create a figure for one sample per class and subject
fig, axs = plt.subplots(len(class_names), len(subject_names), figsize=(12, 10))

for class_index, class_name in enumerate(class_names):
    for subject_index, subject_name in enumerate(subject_names):
        # Filter samples for each class and subject
        class_subject_indices = np.where((labels == class_name) & (subjects == subject_name))[0]

        if len(class_subject_indices) == 0:
            print(f"No samples for class {class_name} and subject {subject_name}.")
            continue

        # Select the first sample for each class-subject pair
        sample_index = class_subject_indices[0]
        ax = axs[class_index, subject_index]

        # Extract and log scale Range-Doppler Map
        rangeDoppler_map = rangeDoppler_data[sample_index]
        log_rangeDoppler_map = np.log10(rangeDoppler_map + 1)

        # Plot Range-Doppler Map
        im = ax.imshow(log_rangeDoppler_map, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xlabel('Range Bins')
        ax.set_ylabel('Doppler Bins')
        ax.set_title(f'{class_name} - {subject_name} - Sample 1')
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

# Adjust layout
plt.tight_layout()
plt.savefig("image.png")
plt.show()

