# Abstract

This project aims to compare the performance between a traditional Machine Learning algorithm (Random Forest) and a Deep Learning model (Convolutional Neural Network) on a big dataset of skin cancer images (HAM10000). The goal is to evaluate their classification accuracy, efficiency, and suitability for medical image analysis.

# Introduction

Skin cancer is one of the most prevalent types of cancer worldwide, and early detection plays a critical role in reducing mortality rates. In this project, we aim to compare the performance of Machine Learning (ML) algorithms and Deep Learning (DL) models for skin cancer classification. Specifically, we will compare the Random Forest (RF) model (ML) and Convolutional Neural Networks (CNNs) (DL) using the HAM10000 dataset.

The objective is to determine which approach provides better accuracy, efficiency, and robustness in classifying skin cancer images. We hypothesize that deep learning will outperform traditional machine learning techniques due to the complex nature of image recognition and feature extraction.

According to recent studies, early detection of skin cancer can significantly increase survival rates, making accurate classification crucial for effective medical interventions.

# Methodology

### Dataset
The dataset used in this project is the **HAM10000 dataset**, which contains 10,000 labeled dermatoscopic images of skin lesions. The data includes various types of skin cancer (e.g., melanoma, basal cell carcinoma, benign keratosis). You can access the dataset directly from [Kaggle's HAM10000 Dataset](https://www.kaggle.com/datasets).

### Data Preprocessing
The following preprocessing steps were performed to prepare the data:
- Resizing all images to a uniform size (e.g., 64x64 pixels).
- Normalizing image pixel values to a range of [0, 1].
- Encoding labels using one-hot encoding for deep learning, and numerical encoding for machine learning.
- Splitting the dataset into training (80%) and testing (20%) sets.

### Model Training
- **Machine Learning (Random Forest)**: We trained a Random Forest classifier to use handcrafted features such as texture, color, and shape.
- **Deep Learning (CNN)**: A Convolutional Neural Network was trained to automatically learn features directly from the image data, leveraging the power of deep learning to capture complex patterns.

### Evaluation Metrics
We used the following evaluation metrics to compare model performance:
- **Accuracy**: The overall percentage of correct predictions.
- **Precision**: The proportion of positive predictions that are actually correct.
- **Recall**: The ability of the model to correctly identify positive cases.
- **F1-Score**: The harmonic mean of precision and recall.
- **AUC-ROC**: The area under the receiver operating characteristic curve.

# Results

After training both the Random Forest (ML) and Convolutional Neural Network (DL) models, the following metrics were recorded:

## Random Forest Model:
- **Accuracy**: 0.74
- **Precision**: 0.69
- **Recall**: 0.74
- **F1-Score**: 0.71
- **AUC-ROC**: 0.79

## CNN Model:
- **Accuracy**: 0.80
- **Precision**: 0.81
- **Recall**: 0.95
- **F1-Score**: 0.87
- **AUC-ROC**: 0.88

The Random Forest model performed with **moderate accuracy and recall**, while the CNN model showed **significantly higher performance** across all metrics, especially in precision and recall.

---

### Model Comparison

Based on the performance metrics, the following comparisons were made:
- **Accuracy**: The CNN model outperformed the Random Forest in terms of accuracy (80% vs 74%).
- **Precision and Recall**: The CNN model achieved higher precision (81% vs 69%) and recall (95% vs 74%), indicating better classification performance, particularly in identifying true positive cases.
- **F1-Score**: The CNN model achieved a much higher F1-Score (0.87 vs 0.71), which demonstrates a better balance between precision and recall.
- **AUC-ROC**: The CNN model had a higher AUC-ROC (0.88 vs 0.79), reflecting better overall model performance in distinguishing between classes.

### Training Time
- **Random Forest**: The Random Forest model was faster to train due to its less complex nature and reliance on manually engineered features.
- **CNN**: The CNN took longer to train due to the complexity of learning features from raw image data, but the results were significantly better.

# Conclusion

This project successfully compared the performance of Machine Learning (Random Forest) and Deep Learning (CNN) for skin cancer classification using the HAM10000 dataset. The results showed that:

- **CNN models** showed a clear advantage over traditional Machine Learning algorithms, especially in terms of **accuracy**, **precision**, and **feature learning**.
- **Random Forest** was faster in training but showed slightly lower accuracy compared to CNN.

In conclusion, CNN-based models are more effective for skin cancer classification, as they can automatically extract features from images, providing higher accuracy in real-world applications.

---

### Future Work

Future improvements could include:
- Experimenting with more advanced architectures like **ResNet** or **Inception**.
- Implementing **ensemble models** that combine the strengths of both Machine Learning and Deep Learning techniques.

# How to Run

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

