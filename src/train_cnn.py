import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from preprocessing import preprocess_data

# تحميل البيانات
X, y, class_names = preprocess_data(img_size=(128, 128))

# تحويل الليبلات إلى one-hot
y_cat = to_categorical(y, num_classes=len(class_names))

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# بناء الموديل CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# تدريب الموديل
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# تقييم الأداء
loss, accuracy = model.evaluate(X_test, y_test)
print(f"CNN Test Accuracy: {accuracy:.4f}")

# التنبؤ واحصائيات مفصلة
y_pred_probs = model.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# حساب AUC
y_test_bin = label_binarize(y_test_labels, classes=range(len(class_names)))
auc_score = roc_auc_score(y_test_bin, y_pred_probs, multi_class='ovo', average='macro')

# حساب باقي المقاييس
cnn_metrics = {
    "Accuracy": accuracy_score(y_test_labels, y_pred_labels),
    "Precision": precision_score(y_test_labels, y_pred_labels, average='macro'),
    "Recall": recall_score(y_test_labels, y_pred_labels, average='macro'),
    "F1-Score": f1_score(y_test_labels, y_pred_labels, average='macro'),
    "AUC-ROC": auc_score
}

# حفظ النتائج في فايل JSON
with open("cnn_metrics.json", "w") as f:
    json.dump(cnn_metrics, f, indent=4)

# حفظ الموديل
model.save("cnn_model.h5")
