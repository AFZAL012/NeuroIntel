import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, RandomFlip, RandomRotation, RandomZoom, RandomTranslation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Define paths (new Kaggle dataset locations)
train_dir = 'Training'
test_dir = 'Testing'
model_save_path = 'models/model.h5'

# Index mapping matching Flask app's expectation (CLASS_LABELS order)
folder_mapping = {
    'pituitary': 0,
    'glioma': 1,
    'notumor': 2,
    'meningioma': 3
}

IMAGE_SIZE = 128

def load_data(data_dir):
    images = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return np.array([]), np.array([])
        
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
if len(X_train) > 0:
    train_indices = np.arange(len(X_train))
    np.random.RandomState(42).shuffle(train_indices)
    X_train, y_train = X_train[train_indices], y_train[train_indices]

print("--- Loading Testing Data ---")
X_test, y_test = load_data(test_dir)
if len(X_test) > 0:
    test_indices = np.arange(len(X_test))
    np.random.RandomState(42).shuffle(test_indices)
    X_test, y_test = X_test[test_indices], y_test[test_indices]

print(f"Train set: {X_train.shape}, labels: {y_train.shape}")
print(f"Test set: {X_test.shape}, labels: {y_test.shape}")

# Build Model using MobileNetV2 with transfer learning and data augmentation
base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze MobileNetV2 base layers

# Data augmentation pipeline (runs only during training phase)
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.15),
    RandomZoom(0.1),
    RandomTranslation(0.1, 0.1)
], name="data_augmentation")

model = Sequential([
    Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    data_augmentation,
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

# Define Callbacks: Early Stopping, Checkpoint, and LR Plateau reduction
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=model_save_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

lr_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

print("--- Training Model ---")
history = model.fit(
    X_train, y_train,
    epochs=15,  # Up to 15 epochs, callbacks will stop training if val_loss plateaus
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, checkpoint, lr_reduction]
)

print("--- Evaluating Model ---")
# Reload the best checkpoint model saved
if os.path.exists(model_save_path):
    print("Loading best weights from checkpoint...")
    model = tf.keras.models.load_model(model_save_path)

# Predict on test data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Generate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*60)
print("RETRAINED MODEL PERFORMANCE METRICS")
print("="*60)
print(f"Accuracy:           {accuracy * 100:.2f}%")
print(f"Precision (Macro):  {precision_macro * 100:.2f}%")
print(f"Recall (Macro):     {recall_macro * 100:.2f}%")
print(f"F1 Score (Macro):   {f1_macro * 100:.2f}%")
print(f"F1 Score (Weighted):{f1_weighted * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['pituitary', 'glioma', 'notumor', 'meningioma']))
print("\nConfusion Matrix:")
print(cm)
print("="*60 + "\n")
