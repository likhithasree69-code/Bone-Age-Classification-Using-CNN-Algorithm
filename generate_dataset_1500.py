import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

dataset_dir = r"c:\Users\Admin\Desktop\Bone Age Classification Using CNN Algorithm\dataset\train"
os.makedirs(dataset_dir, exist_ok=True)

# Generate 1500 images
print("Generating 1500 synthetic human hand X-ray images for training dataset...")

for i in range(1, 1501):
    # Base dark image (224x224)
    img = Image.new('L', (224, 224), color=15) # dark background (15/255 < 0.2)
    draw = ImageDraw.Draw(img)
    
    # Bone colors (bright, > 150)
    bone_color = random.randint(180, 240)
    palm_color = random.randint(130, 170)
    
    # Randomly vary the size/position of bones slightly
    wrist_y = random.randint(170, 190)
    
    # Draw wrist (bottom)
    draw.rectangle([80, wrist_y, 140, 224], fill=bone_color - 10)
    
    # Draw palm
    draw.rectangle([70, 110, 150, wrist_y - 2], fill=palm_color)
    
    # Draw fingers (vertical bars with gaps)
    # Thumb
    draw.line((68, 140, random.randint(35, 50), random.randint(80, 90)), fill=bone_color, width=14)
    
    # Index Finger
    draw.line((85, 110, random.randint(75, 85), random.randint(25, 40)), fill=bone_color, width=13)
    
    # Middle Finger
    draw.line((110, 110, random.randint(105, 115), random.randint(15, 30)), fill=bone_color, width=14)
    
    # Ring Finger
    draw.line((135, 110, random.randint(135, 145), random.randint(25, 40)), fill=bone_color, width=12)
    
    # Pinky Finger
    draw.line((152, 125, random.randint(160, 175), random.randint(55, 75)), fill=bone_color, width=11)
    
    # Add noise (Gaussian)
    noise = np.random.normal(0, random.uniform(5, 12), (224, 224))
    img_arr = np.array(img, dtype=np.float32) + noise
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    
    # Apply a slight blur to make it look like an x-ray glow
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.6, 1.4)))
    
    # Save the image
    age_months = random.randint(24, 216) 
    gender = random.choice(['M', 'F'])
    filename = f"synthetic_xray_{i:04d}_{age_months}m_{gender}.png"
    filepath = os.path.join(dataset_dir, filename)
    img.save(filepath)
    
    if i % 200 == 0:
        print(f"Generated {i}/1500 images...")
        
print("Successfully generated 1500 images in 'dataset/train'!")
