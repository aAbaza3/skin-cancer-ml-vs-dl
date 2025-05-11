import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_data(img_size=(128, 128)):
    print("Preprocessing data...")

    # مسارات الصور والميتا
    img_dir1 = 'data/HAM10000_images_part_1'
    img_dir2 = 'data/HAM10000_images_part_2'
    metadata_path = 'data/HAM10000_metadata.csv'

    # تحميل الميتاداتا
    df = pd.read_csv(metadata_path)
    image_ids = df['image_id']
    labels = df['dx']

    # ترميز الليبلات
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    class_names = le.classes_

    # طباعة التوزيع الأصلي
    print("Original class distribution:")
    print(df['dx'].value_counts())
    print("\nEncoded class distribution:")
    print(pd.Series(y_encoded).value_counts())

    # تحميل الصور وإنشاء X و y
    X = []
    y_final = []
    missing_images = 0

    for img_id, label in zip(image_ids, y_encoded):
        img_path_1 = os.path.join(img_dir1, f"{img_id}.jpg")
        img_path_2 = os.path.join(img_dir2, f"{img_id}.jpg")

        if os.path.exists(img_path_1):
            path = img_path_1
        elif os.path.exists(img_path_2):
            path = img_path_2
        else:
            missing_images += 1
            continue

        img = load_img(path, target_size=img_size)
        img_array = img_to_array(img)
        X.append(img_array)
        y_final.append(label)

    if missing_images:
        print(f"\n {missing_images} images were missing and skipped.")

    X = np.array(X, dtype="float32") / 255.0
    y = np.array(y_final)

    print(f"\n Processed {len(X)} images.")
    return X, y, class_names
