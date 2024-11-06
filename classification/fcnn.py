import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dropout, Dense, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Define parameters
batch_size = 4  # Batch size for training
num_classes = 4  # Number of emotion classes
class_names = ['amusing', 'relaxed', 'boring', 'scary']

# Load preprocessed data from .npz file
data = np.load('drive/MyDrive/processed_data.npz')
features_3 = data['features_3']  # This is the point cloud data (1600, 6)
labels = data['labels']

# Ensure labels are integer-encoded if necessary
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# One-hot encode the integer labels
labels_categorical = to_categorical(labels_encoded, num_classes=num_classes)

# Add a channel dimension to the point cloud data for CNN compatibility
features_3 = np.expand_dims(features_3, -1)  # Shape: (num_samples, 1600, 6, 1)

# Define the CNN model for Feature 3 (Point Cloud Data)
input_shape = (1600, 6, 1)  # Input shape for the point cloud data

# Input layer for Point Cloud Data
input_pcd = Input(shape=input_shape, name="Point_Cloud_Data")

# First Conv Layer
x = Conv2D(32, kernel_size=(8, 2), strides=(2, 1), activation='relu')(input_pcd)

# Second Conv Layer
x = Conv2D(64, kernel_size=(8, 2), strides=(2, 1), activation='relu')(x)

# Third Conv Layer
x = Conv2D(128, kernel_size=(4, 2), strides=(2, 1), activation='relu')(x)

# Fourth Conv Layer
x = Conv2D(256, kernel_size=(4, 1), strides=(2, 1), activation='relu')(x)

# Global Average Pooling and Dropout
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)

# Fully connected layer for embeddings
embedding_output = Dense(128, activation='relu', name='embedding_layer_pcd')(x)  # Embedding layer

# Output layer for classification
output = Dense(num_classes, activation='softmax', name='output_layer')(embedding_output)

# Define the model
pcd_model = Model(inputs=input_pcd, outputs=output)

# Compile the model
pcd_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
pcd_model.summary()

# Train the model on the entire dataset (no train-test split)
history = pcd_model.fit(features_3, labels_categorical, epochs=20, batch_size=batch_size)

# Create an embedding model to extract embeddings from the 'embedding_layer_pcd'
embedding_model_pcd = Model(inputs=pcd_model.input, outputs=pcd_model.get_layer('embedding_layer_pcd').output)

# Generate embeddings for the entire dataset
all_embeddings_pcd = embedding_model_pcd.predict(features_3)

# Save embeddings and labels for the entire dataset to a .npz file
np.savez_compressed('pcd_full_embeddings.npz',
                    embeddings=all_embeddings_pcd,
                    labels=labels_categorical)
np.savez_compressed('drive/MyDrive/pcd_full_embeddings.npz',
                    embeddings=all_embeddings_pcd,
                    labels=labels_categorical)

print("Embeddings for the entire dataset (Point Cloud Data) and labels have been saved to 'pcd_full_embeddings.npz'.")

# Sample evaluation to test model performance on a subset of data
sample_size = int(0.1 * len(features_3))  # Use 10% of data for evaluation
sample_indices = np.random.choice(len(features_3), sample_size, replace=False)
sample_data = features_3[sample_indices]
sample_labels = labels_categorical[sample_indices]

# Evaluate the model on this sample data
sample_loss, sample_accuracy = pcd_model.evaluate(sample_data, sample_labels)
print(f'Sample Evaluation Accuracy (Point Cloud Data): {sample_accuracy * 100:.2f}%')
