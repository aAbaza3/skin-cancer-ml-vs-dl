# Training Random Forest model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data

from evaluate import evaluate_model
import joblib
import os

def train_random_forest():

    # تحميل البيانات المعالجة (الصور والليبلات)
    X, y = preprocess_data() 

    # تسوية الصور (flatten) عشان تبقى 2D
    X_flat = X.reshape((X.shape[0], -1))

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

    # تدريب الموديل
    print("Training Random Forest model...") 
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # التقييم
    evaluate_model(rf, X_test, y_test)

    # حفظ النموذج
    model_path = 'models/random_forest_model.pkl'
    joblib.dump(rf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_random_forest()
