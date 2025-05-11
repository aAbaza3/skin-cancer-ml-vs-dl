from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)
import json
from preprocessing import preprocess_data
import numpy as np

# تحميل البيانات
X, y, class_names = preprocess_data(img_size=(128, 128))
X = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب RF
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred_probs = rf.predict_proba(X_test)

# حساب AUC-ROC
try:
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    auc_score = roc_auc_score(y_test_bin, y_pred_probs, multi_class="ovo", average="macro")
    print(f"RF AUC-ROC (macro): {auc_score:.4f}")
except Exception as e:
    print("AUC-ROC could not be calculated.")
    auc_score = None


# حفظ المقاييس
rf_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average='macro'),
    "Recall": recall_score(y_test, y_pred, average='macro'),
    "F1-Score": f1_score(y_test, y_pred, average='macro'),
    "AUC-ROC": auc_score
}




with open("rf_metrics.json", "w") as f:
    json.dump(rf_metrics, f, indent=4)


