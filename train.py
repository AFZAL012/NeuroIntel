import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.utils import shuffle

# Define paths
train_dir = 'temp_dataset/Training'
test_dir = 'temp_dataset/Testing'
model_save_path = 'models/model.h5'

# Index mapping matching Flask app's expectation
folder_mapping = {
    'pituitary_tumor': 0,
    'glioma_tumor': 1,
    'no_tumor': 2,
    'meningioma_tumor': 3
}

IMAGE_SIZE = 128

def load_data(data_dir):
    images = []
    labels = []
    
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # Check mapping
        if folder_name not in folder_mapping:
            print(f"Skipping unknown folder: {folder_name}")
            continue
            
        label_idx = folder_mapping[folder_name]
        print(f"Loading {folder_name} as label index {label_idx}...")
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_array = np.array(img, dtype=np.float32)
                
                # Preprocess for MobileNetV2
                img_array = preprocess_input(img_array)
                
                images.append(img_array)
                labels.append(label_idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

print("--- Loading Training Data ---")
X_train, y_train = load_data(train_dir)
X_train, y_train = shuffle(X_train, y_train, random_state=42)

print("--- Loading Testing Data ---")
X_test, y_test = load_data(test_dir)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

print(f"Train set: {X_train.shape}, labels: {y_train.shape}")
print(f"Test set: {X_test.shape}, labels: {y_test.shape}")

# Build Model using MobileNetV2
base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze MobileNetV2 base layers

model = Sequential([
    Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

model.summary()

print("--- Training Model ---")
history = model.fit(
    X_train, y_train,
    epochs=4,
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("--- Evaluating Model ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Save the trained model
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"Model saved successfully to {model_save_path}")
