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

IMG_SIZE = (224, 224)

def parse_label(filename):
    """Extract age in months from the filename."""
    # Pattern to match 'Nm' where N is digit(s)
    match = re.search(r'_(\d+)m_', filename)
    if match:
        return float(match.group(1))
    return None

def load_data(directory, limit=None):
    images = []
    labels = []
    
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    if limit:
        files = files[:limit]
        
    print(f"Loading {len(files)} images from {directory}...")
    
    for filename in files:
        label = parse_label(filename)
        if label is None:
            continue
            
        img_path = os.path.join(directory, filename)
        try:
            # Grayscale for X-rays
            img = Image.open(img_path).convert('L')
            img = img.resize(IMG_SIZE, Image.LANCZOS)
            img_array = np.array(img, dtype='float32') / 255.0
            
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            
    # Reshape for CNN input: (N, 224, 224, 1)
    if images:
        images_np = np.stack(images, axis=0)
        images_np = np.expand_dims(images_np, axis=-1)
        return images_np, np.array(labels, dtype='float32')
    else:
        return None, None

def build_model():
    """Robust CNN for Regression with Data Augmentation."""
    
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.1, fill_mode='constant', fill_value=0),
        layers.RandomZoom(0.1, fill_mode='constant', fill_value=0),
        layers.RandomContrast(0.2), # Good for different xray intensities
        layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=0)
    ])
    
    model = tf.keras.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        # Apply augmentation only during training
        data_augmentation,
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.GlobalAveragePooling2D(), # Better than Flatten for robustness
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4), # Increased for more diversity
        layers.Dense(1) # Final output: month age
    ])
    
    # Using a slightly lower learning rate for stable convergence on regr
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    import traceback
    try:
        print(f"Data Dir Check: {TRAIN_DIR} exists? {os.path.exists(TRAIN_DIR)}")
        # Load all data
        train_x, train_y = load_data(TRAIN_DIR)
        val_x, val_y = load_data(VAL_DIR)
        
        if train_x is not None:
            print(f"Loaded {train_x.shape[0]} training images.")
            print("Starting training with Augmented CNN...")
            model = build_model()
            
            # Early stopping to prevent overfitting
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            history = model.fit(
                train_x, train_y,
                epochs=40, # increased from 20 for full dataset
                batch_size=32,
                validation_data=(val_x, val_y) if val_x is not None else None,
                callbacks=[early_stop],
                verbose=1
            )
            
            # Save model in .h5 format as expected by predict.py
            model.save(MODEL_SAVE_PATH)
            print(f"Enhanced Model saved to: {MODEL_SAVE_PATH}")
        else:
            print("No training data found.")
    except Exception as e:
        print("Training failed with error:")
        traceback.print_exc()
