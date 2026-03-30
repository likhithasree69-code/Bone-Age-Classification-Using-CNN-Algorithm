import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import re

# Data path
DATASET_DIR = r"c:\Users\Admin\Desktop\Bone Age Classification Using CNN Algorithm\dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')
MODEL_SAVE_PATH = r"c:\Users\Admin\Desktop\Bone Age Classification Using CNN Algorithm\classifier\ml_model\trained_model.h5"

IMG_SIZE = (128, 128) # Smaller for speed

def parse_label(filename):
    match = re.search(r'_(\d+)m_', filename)
    if match:
        return float(match.group(1))
    return None

def load_data(directory, limit=500):
    images = []
    labels = []
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    files = files[:limit]
    
    print(f"Loading {len(files)} images from {directory}...")
    for filename in files:
        label = parse_label(filename)
        if label is None: continue
        img_path = os.path.join(directory, filename)
        try:
            img = Image.open(img_path).convert('L')
            img = img.resize(IMG_SIZE, Image.LANCZOS)
            img_array = np.array(img, dtype='float32') / 255.0
            images.append(img_array)
            labels.append(label)
        except: continue
            
    if images:
        images_np = np.stack(images, axis=0)
        images_np = np.expand_dims(images_np, axis=-1)
        return images_np, np.array(labels, dtype='float32')
    return None, None

def build_lite_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1) # monthly age
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    # Load 500 images for quick training
    train_x, train_y = load_data(TRAIN_DIR, limit=800)
    val_x, val_y = load_data(VAL_DIR, limit=200)
    
    if train_x is not None:
        print(f"Loaded {train_x.shape[0]} images. Starting LITE training...")
        model = build_lite_model()
        model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(val_x, val_y), verbose=1)
        model.save(MODEL_SAVE_PATH)
        print(f"Lite Model saved to: {MODEL_SAVE_PATH}")
    else:
        print("No training data found.")
