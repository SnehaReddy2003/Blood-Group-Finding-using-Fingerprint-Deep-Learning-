import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define dataset path
dataset_path = r"D:\SAP\SAP project\dataset"
blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
label_map = {group: idx for idx, group in enumerate(blood_groups)}

# Load images and labels
images = []
labels = []
for group in blood_groups:
    folder_path = os.path.join(dataset_path, group)
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist. Skipping.")
        continue
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label_map[group])
        else:
            print(f"Warning: Could not load image {img_path}")

# Convert to numpy arrays
if len(images) == 0:
    raise ValueError("No images were loaded. Check dataset path and contents.")
images = np.array(images).reshape(-1, 128, 128, 1) / 255.0  # Normalize
labels = to_categorical(np.array(labels), num_classes=len(blood_groups))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(blood_groups), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Training completed.")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the model
model.save("blood_group_model.h5")
print("Model saved successfully as 'blood_group_model.h5'")