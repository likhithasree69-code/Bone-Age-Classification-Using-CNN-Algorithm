import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFilter

# Source base realistic x-ray images
base_images = [
    r"C:\Users\Admin\.gemini\antigravity\brain\be1f7d4b-c25a-4bba-a19f-4ad1ab09f36d\real_hand_xray_1_1773566321999.png",
    r"C:\Users\Admin\.gemini\antigravity\brain\be1f7d4b-c25a-4bba-a19f-4ad1ab09f36d\real_hand_xray_2_1773566399786.png"
]

dataset_dir = r"c:\Users\Admin\Desktop\Bone Age Classification Using CNN Algorithm\dataset\train"

# Ensure directory exists but DO NOT DELETE existing files
os.makedirs(dataset_dir, exist_ok=True)

loaded_base_images = [Image.open(img_path).convert('L') for img_path in base_images if os.path.exists(img_path)]

if not loaded_base_images:
    print("Error: Could not find the base realistic images.")
    exit()

print("Adding 500 more realistic X-rays with wider age range (1-240 months)...")

def add_abnormality(img, severity):
    """Draws realistic looking fractures/medical issues on the x-ray."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    if severity == "None":
        return img
        
    num_fractures = 1
    if severity == "Medium":
        num_fractures = 2
    elif severity == "High":
        num_fractures = 3
        
    for _ in range(num_fractures):
        x = random.randint(int(w*0.3), int(w*0.7))
        y = random.randint(int(h*0.4), int(h*0.8))
        
        line_len = random.randint(15, 30) if severity == "Low" else random.randint(30, 60)
        thickness = random.randint(1, 2) if severity == "Low" else random.randint(2, 4)
        
        points = [(x, y)]
        curr_x, curr_y = x, y
        for _ in range(5):
            curr_x += random.randint(-5, 5)
            curr_y += random.randint(5, 12)
            points.append((curr_x, curr_y))
            
        draw.line(points, fill=40, width=thickness)
        
        if severity == "High":
            overlay = Image.new('L', img.size, 0)
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.ellipse([x-25, y-15, x+25, y+35], fill=180)
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=15))
            img.paste(ImageOps.colorize(overlay, (0,0,0), (255,255,255)).convert('L'), (0,0), mask=overlay)

    return img

# Start from index 1501
for i in range(1501, 2001):
    base_img = random.choice(loaded_base_images).copy()
    
    chance = random.random()
    if chance < 0.2:
        severity = "None"
    elif chance < 0.5:
        severity = "Low"
    elif chance < 0.8:
        severity = "Medium"
    else:
        severity = "High"
        
    img = add_abnormality(base_img, severity)
    
    # Random Rotation
    angle = random.uniform(-10.0, 10.0)
    img = img.rotate(angle, fillcolor=0)
    
    # Zoom/Crop
    zoom_factor = random.uniform(1.0, 1.2)
    w, h = img.size
    cw, ch = int(w/zoom_factor), int(h/zoom_factor)
    left = random.randint(0, w - cw)
    top = random.randint(0, h - ch)
    img = img.crop((left, top, left + cw, top + ch))
    img = img.resize((w, h), Image.LANCZOS)
    
    # Brightness & Contrast
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.9, 1.1))
    
    # Horizontal Flip
    if random.choice([True, False]):
        img = ImageOps.mirror(img)
        
    # Gaussian noise
    img_arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(2, 5), img_arr.shape)
    img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    
    # Wider age range: 1 month to 240 months (0 to 20 years)
    age_months = random.randint(1, 240) 
    gender = random.choice(['M', 'F'])
    
    filename = f"real_xray_{i:04d}_Sev-{severity}_{age_months}m_{gender}.png"
    filepath = os.path.join(dataset_dir, filename)
    
    # Resize to standard model size 224x224
    img = img.resize((224, 224), Image.LANCZOS)
    img.save(filepath)
    
    if i % 100 == 0:
        print(f"Added {i-1500}/500 more images...")

print(f"Successfully added 500 more realistic X-rays in '{dataset_dir}' with age range 1-240m.")
