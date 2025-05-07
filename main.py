from train_rf import train_random_forest
from train_cnn import train_cnn

def main():
    print("== Training Random Forest ==")
    train_random_forest()

    print("== Training CNN ==")
    train_cnn()

if __name__ == "__main__":
    main()
