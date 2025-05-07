import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from preprocessing import preprocess_data

def get_metrics(model, X_test, y_test, is_dl=False):
    y_pred = model.predict(X_test)
    
    if is_dl:
        y_pred = np.argmax(y_pred, axis=1)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

def compare_models():
    # تحميل البيانات
    X, y = preprocess_data()
    X_flat = X.reshape((X.shape[0], -1))

    # تقسيم البيانات
    X_train_flat, X_test_flat, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # تقييم Random Forest
    rf = joblib.load("models/random_forest_model.pkl")
    results['Random Forest'] = get_metrics(rf, X_test_flat, y_test)

    # تقييم CNN
    cnn = load_model("models/cnn_model.h5")
    results['CNN'] = get_metrics(cnn, X_test, y_test, is_dl=True)

    # تحويل النتائج إلى DataFrame
    df_results = pd.DataFrame(results).T
    df_results.to_csv("comparison_results.csv")
    print("\nSaved results to comparison_results.csv")

    # رسم Bar Chart
    df_results.plot(kind='bar', figsize=(10, 6))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("comparison_chart.png")
    print("Saved chart as comparison_chart.png")
    plt.show()

if __name__ == "__main__":
    compare_models()
