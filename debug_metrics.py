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
        bone_mean = np.mean(arr[arr > 0.18])
        grad_x = np.diff(arr, axis=1)
        grad_y = np.diff(arr, axis=0)
        edge_density = (np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y))) / 2
        ratio = edge_density / (bone_mean + 1e-6)
        print(f"File: {os.path.basename(path)}")
        print(f"  Bone Mean: {bone_mean:.4f}")
        print(f"  Edge Density: {edge_density:.4f}")
        print(f"  Ratio: {ratio:.4f}")
        print("-" * 20)
