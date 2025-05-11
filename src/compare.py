import json

# تحميل النتائج
with open("cnn_metrics.json", "r") as f:
    cnn_metrics = json.load(f)

with open("rf_metrics.json", "r") as f:
    rf_metrics = json.load(f)

# طباعة مقارنة بين كل مقياس
print("📊 Comparison Between CNN and Random Forest\n")
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

for metric in metrics:
    cnn_value = cnn_metrics.get(metric, None)
    rf_value = rf_metrics.get(metric, None)

    if cnn_value is None or rf_value is None:
        print(f"{metric}: ⚠️ Missing value in one of the models")
        continue

    better = "CNN" if cnn_value > rf_value else "Random Forest" if rf_value > cnn_value else "Equal"
    
    print(f"{metric}:")
    print(f"  - CNN:           {cnn_value:.4f}")
    print(f"  - Random Forest: {rf_value:.4f}")
    print(f"  ➤ Better: {better}\n")
