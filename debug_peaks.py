import numpy as np
from PIL import Image
import os

base_images = [
    r"C:\Users\Admin\.gemini\antigravity\brain\c8c8c68e-398d-4b17-b351-f99ff5138ba8\realistic_hand_xray_child_5y_1774256316687.png",
    r"C:\Users\Admin\.gemini\antigravity\brain\c8c8c68e-398d-4b17-b351-f99ff5138ba8\realistic_hand_xray_child_10y_1774256199109.png",
    r"C:\Users\Admin\.gemini\antigravity\brain\c8c8c68e-398d-4b17-b351-f99ff5138ba8\realistic_hand_xray_adult_40y_1774256244473.png",
    r"C:\Users\Admin\.gemini\antigravity\brain\c8c8c68e-398d-4b17-b351-f99ff5138ba8\realistic_hand_xray_elderly_60y_1774256276073.png"
]

for path in base_images:
    if os.path.exists(path):
        img = Image.open(path).convert('L').resize((224, 224))
        arr = np.array(img) / 255.0
        peak = np.percentile(arr, 95)
        print(f"File: {os.path.basename(path)}")
        print(f"  Peak (95th): {peak:.4f}")
        print("-" * 20)
