import os
import base64
import random
import numpy as np
from PIL import Image, ImageEnhance

def encode_image_to_data_url(img_path: str) -> str:
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(img_path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{img_b64}"

def safe_join(folder: str, filename: str) -> str:
    return os.path.join(folder, filename)

def rotate_image(img_path: str, save_dir: str):
    """ICL4: image rotation"""
    angle = random.choice([90, 180, 270])
    img = Image.open(img_path)
    rotated = img.rotate(angle, expand=True)
    
    name, ext = os.path.splitext(os.path.basename(img_path))
    new_name = f"{name}_rot{angle}.jpg"
    save_path = os.path.join(save_dir, new_name)
    rotated.save(save_path)
    return save_path, f"rot{angle}"

def augment_image(img_path: str, save_path: str):
    """ICL5: Image comprehensive enhancement"""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    logs = []

    angle = random.choice([90, 180, 270])
    img = img.rotate(angle, expand=True)
    logs.append(f"rot={angle}")

    scale = random.uniform(0.8, 1.2)
    img = img.resize((int(w * scale), int(h * scale)))
    logs.append(f"scl={scale:.2f}")

    brightness = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    logs.append(f"brt={brightness:.2f}")

    contrast = random.uniform(0.7, 1.3)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    logs.append(f"con={contrast:.2f}")

    noise_level = random.uniform(0.0, 0.03)
    np_img = np.array(img) / 255.0
    noise = np.random.normal(0, noise_level, np_img.shape)
    np_img = np.clip(np_img + noise, 0, 1)
    img = Image.fromarray((np_img * 255).astype(np.uint8))
    logs.append(f"nse={noise_level:.3f}")

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        logs.append("flip=H")

    img.save(save_path)
    return save_path, ", ".join(logs)