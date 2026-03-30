import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFilter

# Source base realistic x-ray images
base_images = [
    r"C:\Users\Admin\.gemini\antigravity\brain\be1f7d4b-c25a-4bba-a19f-4ad1ab09f36d\real_hand_xray_1_1773566321999.png",
    r"C:\Users\Admin\.gemini\antigravity\brain\be1f7d4b-c25a-4bba-a19f-4ad1ab09f36d\real_hand_xray_2_1773566399786.png"
]

def add_abnormality(img, severity):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    if severity == "None": return img
    num_fractures = 1 if severity == "Low" else (2 if severity == "Medium" else 3)
    for _ in range(num_fractures):
        x = random.randint(int(w*0.3), int(w*0.7))
        y = random.randint(int(h*0.4), int(h*0.8))
        line_len = random.randint(15, 30) if severity == "Low" else random.randint(30, 60)
        thickness = random.randint(1, 2) if severity == "Low" else random.randint(2, 4)
        points = [(x, y)]
        curr_x, curr_y = x, y
        for _ in range(5):
            curr_x += random.randint(-5, 5); curr_y += random.randint(5, 12)
            points.append((curr_x, curr_y))
        draw.line(points, fill=40, width=thickness)
        if severity == "High":
            overlay = Image.new('L', img.size, 0)
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.ellipse([x-25, y-15, x+25, y+35], fill=180)
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=15))
            img.paste(ImageOps.colorize(overlay, (0,0,0), (255,255,255)).convert('L'), (0,0), mask=overlay)
    return img

def create_images(dir_path, count, start_id, min_age_years, max_age_years):
    os.makedirs(dir_path, exist_ok=True)
    # Ensure base images exist
    loaded_base_images = [Image.open(img_path).convert('L') for img_path in base_images if os.path.exists(img_path)]
    if not loaded_base_images:
        print(f"Error: Could not find base images for {dir_path}")
        return
        
    for i in range(start_id, start_id + count):
        base_img = random.choice(loaded_base_images).copy()
        chance = random.random()
        severity = "None" if chance < 0.2 else ("Low" if chance < 0.5 else ("Medium" if chance < 0.8 else "High"))
        img = add_abnormality(base_img, severity)
        img = img.rotate(random.uniform(-10.0, 10.0), fillcolor=0)
        zoom_factor = random.uniform(1.0, 1.2)
        w, h = img.size
        cw, ch = int(w/zoom_factor), int(h/zoom_factor)
        left = random.randint(0, w - cw); top = random.randint(0, h - ch)
        img = img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
        if random.choice([True, False]): img = ImageOps.mirror(img)
        img_arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(2, 5), img_arr.shape)
        img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        
        # Age range: 30 to 60 years (360 to 720 months)
        age_months = random.randint(min_age_years * 12, max_age_years * 12) 
        gender = random.choice(['M', 'F'])
        filename = f"real_xray_{i:04d}_Sev-{severity}_{age_months}m_{gender}.png"
        img.resize((224, 224), Image.LANCZOS).save(os.path.join(dir_path, filename))

train_dir = r"c:\Users\Admin\Desktop\Bone Age Classification Using CNN Algorithm\dataset\train"
test_dir = r"c:\Users\Admin\Desktop\Bone Age Classification Using CNN Algorithm\dataset\test"
val_dir = r"c:\Users\Admin\Desktop\Bone Age Classification Using CNN Algorithm\dataset\val"

print("Adding 300 training, 50 test, and 50 validation images for age group 30-60 years...")
create_images(train_dir, 300, 2201, 30, 60)
create_images(test_dir, 50, 2501, 30, 60)
create_images(val_dir, 50, 2551, 30, 60)
print("Successfully added adult X-ray images (30-60 years) to the dataset.")
