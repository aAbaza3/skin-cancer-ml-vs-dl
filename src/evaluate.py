from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test, is_dl=False):
    if is_dl:
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)  # تحويل الاحتمالات إلى صنف class
    else:
        y_pred = model.predict(X_test)
        
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # رسم المصفوفة الارتباكية (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
