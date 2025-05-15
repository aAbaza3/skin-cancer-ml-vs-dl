from src.train_rf import train_random_forest
from src.train_cnn import train_cnn

def main():
    print("== Training Random Forest ==")
    rf_acc, rf_report = train_random_forest()
    print(f"\nRandom Forest Accuracy: {rf_acc:.4f}")
    print("Random Forest Classification Report:\n")
    print(rf_report)

    print("== Training CNN ==")
    cnn_acc, cnn_report = train_cnn()
    print(f"\nCNN Accuracy: {cnn_acc:.4f}")
    print("CNN Classification Report:\n")
    print(cnn_report)

    # مقارنة مباشرة
    print("== Model Comparison ==")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"CNN Accuracy         : {cnn_acc:.4f}")

    if cnn_acc > rf_acc:
        print("CNN performed better.")
    elif rf_acc > cnn_acc:
        print("Random Forest performed better.")
    else:
        print("Both models performed equally.")

if __name__ == "__main__":
    main()
