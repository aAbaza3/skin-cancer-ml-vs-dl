# Training CNN model
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from preprocessing import preprocess_data
from evaluate import evaluate_model

def train_cnn():
    # تحميل البيانات
    X, y = preprocess_data()

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = X_train.shape[1:]

    # بناء النموذج
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(set(y)), activation='softmax')  # عدد الفئات
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # إيقاف مبكر للتدريب
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # تدريب النموذج
    print("Training CNN model...")
    model.fit(X_train, y_train, 
              validation_split=0.2,
              epochs=15,
              batch_size=32,
              callbacks=[early_stop],
              verbose=1)

    # التقييم
    evaluate_model(model, X_test, y_test, is_dl=True)
    # حفظ النموذج
    model.save("models/cnn_model.h5")
    print("CNN model saved to models/cnn_model.h5")


if __name__ == "__main__":
    train_cnn()

