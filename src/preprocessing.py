import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_data(img_dir='data/HAM10000_images_part_1', metadata_path='data/HAM10000_metadata.csv', img_size=(64, 64)):
    print("Preprocessing data...")

    # قراءة الميتاداتا
    df = pd.read_csv(metadata_path)
    image_ids = df['image_id']
    labels = df['dx']

    # ترميز الليبلات (text → numeric)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # تحميل وتحويل الصور إلى arrays
    X = []
    missing_images = 0

    for img_id in image_ids:
        img_path_1 = os.path.join('data', 'HAM10000_images_part_1', f"{img_id}.jpg")
        img_path_2 = os.path.join('data', 'HAM10000_images_part_2', f"{img_id}.jpg")

        if os.path.exists(img_path_1):
            path = img_path_1
        elif os.path.exists(img_path_2):
            path = img_path_2
        else:
            missing_images += 1
            continue

        # تحميل الصورة وتحجيمها وتحويلها لـ array
        img = load_img(path, target_size=img_size)
        img_array = img_to_array(img)
        X.append(img_array)

    if missing_images:
        print(f"{missing_images} images were missing and skipped.")

    # تحويل X و y إلى NumPy arrays
    X = np.array(X, dtype="float32") / 255.0  # Normalize
    y = np.array(y)

    print(f"Processed {len(X)} images.")
    return X, y
