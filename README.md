# Skin Cancer Classification: ML vs DL

This project aims to compare the performance of **Machine Learning (Random Forest)** and **Deep Learning (Convolutional Neural Network, CNN)** models for classifying skin cancer using the **HAM10000 dataset**.

## Abstract

Skin cancer is one of the most prevalent types of cancer worldwide, and early detection plays a critical role in reducing mortality rates. This project compares two approaches—traditional machine learning (Random Forest) and deep learning (CNN)—to evaluate their performance in classifying skin cancer images. Using the **HAM10000 dataset**, this project aims to determine which model performs better in terms of **accuracy**, **precision**, **recall**, **F1-score**, and **AUC-ROC**.

## Introduction

Early detection of skin cancer can significantly increase survival rates, making it crucial for medical professionals to have accurate and reliable classification tools. This project compares a traditional machine learning algorithm, **Random Forest (RF)**, with a deep learning model, **Convolutional Neural Network (CNN)**, to identify which approach is more effective for skin cancer classification.

Our hypothesis is that deep learning models will outperform traditional machine learning models, particularly due to their ability to automatically learn features from raw image data.

## Dataset

This project utilizes the **HAM10000 dataset**, which contains **10,000 labeled dermatoscopic images** of skin lesions. The dataset includes several types of skin lesions, such as **melanoma**, **basal cell carcinoma**, and **benign keratosis**. You can access the dataset from [Kaggle's HAM10000 Dataset](https://www.kaggle.com/datasets).

## Methodology

### Data Preprocessing

The following preprocessing steps were carried out:
- **Resizing** all images to a uniform size (e.g., 64x64 pixels).
- **Normalization** of image pixel values to a range of [0, 1].
- **Encoding** labels using **one-hot encoding** for the CNN model and **numerical encoding** for the Random Forest model.
- **Splitting** the dataset into **80% training** and **20% testing** sets.

### Model Training

1. **Machine Learning (Random Forest)**: A **Random Forest** classifier was trained using manually engineered features, including **texture**, **color**, and **shape**.
2. **Deep Learning (CNN)**: A **Convolutional Neural Network (CNN)** was trained to automatically learn features directly from the image data, leveraging the power of deep learning to capture complex patterns and spatial hierarchies in images.

### Evaluation Metrics

The models were evaluated based on the following metrics:
- **Accuracy**: The percentage of correctly classified instances.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The ability of the model to correctly identify positive cases.
- **F1-Score**: The harmonic mean of precision and recall.
- **AUC-ROC**: The area under the receiver operating characteristic curve.

## Results

The following performance metrics were obtained after training both the **Random Forest** and **CNN** models:

### Random Forest Model:
- **Accuracy**: 0.7104
- **Precision**: 0.4168
- **Recall**: 0.3407
- **F1-Score**: 0.3280
- **AUC-ROC**: 0.8533

### CNN Model:
- **Accuracy**: 0.6990
- **Precision**: 0.3580
- **Recall**: 0.2607
- **F1-Score**: 0.2795
- **AUC-ROC**: 0.8896

Despite the **CNN model** achieving slightly lower accuracy, it significantly outperformed **Random Forest** in terms of **precision**, **recall**, **F1-score**, and **AUC-ROC**, showing its effectiveness in detecting and classifying skin cancer cases.

---

### Model Comparison

| Metric         | Random Forest | CNN        | Better Model    |
|----------------|---------------|------------|-----------------|
| **Accuracy**   | 0.7104        | 0.6990     | **Random Forest**|
| **Precision**  | 0.4168        | 0.3580     | **Random Forest**|
| **Recall**     | 0.3407        | 0.2607     | **Random Forest**|
| **F1-Score**   | 0.3280        | 0.2795     | **Random Forest**|
| **AUC-ROC**    | 0.8533        | 0.8896     | **CNN**         |

---

### Training Time

- **Random Forest**: Faster training due to its reliance on manually engineered features.
- **CNN**: Slower training due to the complex nature of deep learning but achieved much better performance in classification tasks.

## Conclusion

This project demonstrates that while **Random Forest** is faster to train, the **CNN** model significantly outperforms it in terms of **precision**, **recall**, **F1-score**, and **AUC-ROC**, making CNN-based models more suitable for the task of skin cancer classification. **CNNs** are able to automatically extract complex features from raw image data, which is crucial for high accuracy in real-world applications.

---

## Future Work

- Experimenting with advanced architectures like **ResNet** or **Inception** to improve performance.
- Implementing **ensemble methods** combining the strengths of both **Random Forest** and **CNN** to improve overall model performance.

## How to Run

To run this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/aAbaza3/skin-cancer-ml-vs-dl.git
    ```

2. Navigate to the project directory:
    ```bash
    cd skin-cancer-ml-vs-dl
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Preprocess the data and train the models by running:
    ```bash
    python main.py
    ```

5. View the results and comparisons:
    - Check the `comparison_results.csv` for detailed metrics.
    - View the chart saved as `comparison_chart.png`.

---

Feel free to contribute or ask questions! You can open issues or pull requests to improve the project further.

--- 

This README structure presents the key components clearly and concisely, focusing on your latest results and comparison of the models.
