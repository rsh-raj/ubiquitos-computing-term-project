import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = np.load('drive/MyDrive/processed_data_sub.npz')
rangeDoppler_data = data['features_2']  # Assuming 'features_2' is the range-doppler map with shape (num_samples, 182, 256)
labels = data['labels']  # Assuming labels correspond to classes like 'amusing', 'relaxed', etc.
subjects = data['subjects']  # Assuming 'subjects' contains the subject names like 'astitva', 'pranil', 'rishi'
# Load the processed data
# data = np.load('drive/MyDrive/processed_data.npz')
pcd_data = data['features_3']  # Assuming 'features_3' is the point cloud data with shape (num_samples, 1600, 6)
# labels = data['labels']  # Class labels for each sample
# subjects = data['subjects']  # Subject labels for each sample

# Define class and subject names for visualization
class_names = ['amusing', 'relaxed', 'boring', 'scary']
subject_names = ['astitva', 'pranil', 'rishi']
num_samples_to_plot = 1  # Number of samples to visualize per class-subject pair

# Create a figure for plotting
fig = plt.figure(figsize=(18, 12))

# Iterate over each class and subject
plot_index = 1
for class_index, class_name in enumerate(class_names):
    for subject_index, subject_name in enumerate(subject_names):
        # Filter samples belonging to the current class and subject
        class_subject_indices = np.where((labels == class_name) & (subjects == subject_name))[0]

        # Ensure we have samples to plot
        if len(class_subject_indices) < num_samples_to_plot:
            print(f"Not enough samples for class {class_name} and subject {subject_name}. Found only {len(class_subject_indices)} samples.")
            continue

        # Select a sample for visualization
        sample_index = class_subject_indices[0]
        point_cloud = pcd_data[sample_index]  # Shape: (1600, 6)

        # Separate x, y, z, and velocity values
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        velocity = point_cloud[:, 3]

        # Create a 3D scatter plot
        ax = fig.add_subplot(len(class_names), len(subject_names), plot_index, projection='3d')
        scatter = ax.scatter(x, y, z, c=velocity, cmap='viridis', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{class_name} - {subject_name} - Sample {plot_index}')

        # Add color bar for velocity
        fig.colorbar(scatter, ax=ax, shrink=0.6, label='Velocity')

        plot_index += 1

# Adjust layout and show plot
plt.tight_layout()
plt.show()
