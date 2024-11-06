import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dropout, Dense, Input, TimeDistributed, Flatten
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define parameters
stack_size = 5  # Number of consecutive frames to stack
batch_size = 4  # Batch size for training
num_classes = 4  # Number of emotion classes
class_names = ['amusing', 'relaxed', 'boring', 'scary']

# Load preprocessed data from .npz file
data = np.load('drive/MyDrive/processed_data.npz')
features_1 = data['features_1']
features_2 = data['features_2']
labels = data['labels']

# Combine Feature 1 and Feature 2 to prepare for CNN input
feat1_data_expanded = np.expand_dims(features_1, axis=1)  # Shape (num_samples, 1, 256)
feat2_data_combined = np.concatenate((feat1_data_expanded, features_2), axis=1)  # Shape: (num_samples, 183, 256)

# Ensure labels are integer-encoded if necessary
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# One-hot encode the integer labels
labels_categorical = to_categorical(labels_encoded, num_classes=num_classes)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(feat2_data_combined, labels_categorical, test_size=0.3, random_state=42)

del data
del features_1
del features_2
del labels
del labels_encoded
# Define the data generator for frame stacking
class FrameStackingGenerator(Sequence):
    def __init__(self, data, labels, stack_size, batch_size):
        self.data = data
        self.labels = labels
        self.stack_size = stack_size
        self.batch_size = batch_size
        self.indices = np.arange(data.shape[0] - stack_size + 1)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        # Select batch of indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Prepare batch data
        X_batch = np.array([self.data[i:i + self.stack_size] for i in batch_indices])
        y_batch = self.labels[batch_indices + self.stack_size - 1]  # Use label of the last frame in the stack

        # Expand dimensions for CNN input
        X_batch = np.expand_dims(X_batch, -1)  # Shape: (batch_size, stack_size, 183, 256, 1)

        return X_batch, y_batch

# Instantiate the data generators
train_generator = FrameStackingGenerator(X_train, y_train, stack_size, batch_size)
test_generator = FrameStackingGenerator(X_test, y_test, stack_size, batch_size)

# Define the CNN model with frame stacking
model = Sequential([
    Input(shape=(stack_size, 183, 256, 1)),  # Input shape includes stack_size for frame stacking
    TimeDistributed(Conv2D(32, (8, 8), strides=(1, 1), activation='relu')),
    TimeDistributed(Conv2D(64, (8, 8), strides=(1, 1), activation='relu')),
    TimeDistributed(Conv2D(128, (4, 4), strides=(1, 1), activation='relu')),
    TimeDistributed(GlobalAveragePooling2D()),
    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu', name='embedding_layer'),  # Named layer for embeddings
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model using the generator
history = model.fit(train_generator, epochs=20, validation_data=test_generator)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Create an embedding model to extract embeddings from the dense layer before the final classification
embedding_model = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)

# Generate embeddings for the training and test sets
train_embeddings = embedding_model.predict(train_generator)
test_embeddings = embedding_model.predict(test_generator)

# Save embeddings and labels to a .npz file
np.savez_compressed('cnn_embeddings.npz',
                    train_embeddings=train_embeddings,
                    test_embeddings=test_embeddings,
                    train_labels=y_train,
                    test_labels=y_test)

print("Embeddings and labels have been saved to 'cnn_embeddings.npz'")

# Example of loading embeddings for use in a classifier
loaded_data = np.load('cnn_embeddings.npz')
train_embeddings = loaded_data['train_embeddings']
test_embeddings = loaded_data['test_embeddings']
train_labels = loaded_data['train_labels']
test_labels = loaded_data['test_labels']

print("Loaded embeddings and labels for further use in a classifier.")
