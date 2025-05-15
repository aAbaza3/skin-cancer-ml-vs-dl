---

# Skin Cancer Classification: Machine Learning vs Deep Learning

## Abstract

This project compares the performance of a traditional Machine Learning algorithm (Random Forest) and a Deep Learning model (Convolutional Neural Network) on a large dataset of skin cancer images (HAM10000). The main goal is to evaluate and contrast their classification accuracy, efficiency, and suitability for medical image analysis.

---

## Introduction

Skin cancer is one of the most common cancers globally. Early and accurate detection is essential to improve survival rates and guide treatment. This project investigates two approaches for classifying skin lesion images:

* **Random Forest (RF)**: A traditional machine learning algorithm.
* **Convolutional Neural Network (CNN)**: A deep learning model capable of extracting complex features from images.

We use the **HAM10000** dataset to test and evaluate both models. Our objective is to determine which technique yields better accuracy and is more practical for real-world medical use.

---

## Methodology

### Dataset

The dataset used is the [HAM10000 dataset](https://www.kaggle.com/datasets), containing **10,000 dermatoscopic images** labeled with seven types of skin lesions (e.g., melanoma, nevus, BCC, etc.).

### Preprocessing

* All images resized to 64x64 pixels.
* Pixel normalization to range \[0, 1].
* Label encoding (one-hot for CNN, integer labels for RF).
* Dataset split: 80% training, 20% testing.

### Models

* **Random Forest**: Trained on manually extracted features (e.g., color histograms, texture).
* **CNN**: Trained directly on raw image data using several convolutional and pooling layers.

### Evaluation Metrics

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Support**
* **Macro average** and **Weighted average**
=======
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
>>>>>>> c372d3d25e08c5bf2d1ca02e423965bf54a02213

---

## Results

<<<<<<< HEAD
### 📊 CNN Model

* **Accuracy**: `0.7309`
* **Macro Avg F1-Score**: `0.42`
* **Weighted Avg F1-Score**: `0.71`

#### Class-wise Performance:

| Class | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| akiec | 0.46      | 0.26   | 0.33     |
| bcc   | 0.46      | 0.40   | 0.42     |
| bkl   | 0.43      | 0.44   | 0.44     |
| df    | 0.00      | 0.00   | 0.00     |
| mel   | 0.43      | 0.26   | 0.33     |
| nv    | 0.82      | 0.92   | 0.87     |
| vasc  | 0.85      | 0.39   | 0.54     |

---

### 🌳 Random Forest Model

* **Accuracy**: `0.7164`
* **Macro Avg F1-Score**: `0.29`
* **Weighted Avg F1-Score**: `0.66`

#### Class-wise Performance:
=======
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
>>>>>>> c372d3d25e08c5bf2d1ca02e423965bf54a02213

| Class | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| akiec | 0.33      | 0.17   | 0.22     |
| bcc   | 0.47      | 0.26   | 0.34     |
| bkl   | 0.44      | 0.32   | 0.37     |
| df    | 0.00      | 0.00   | 0.00     |
| mel   | 0.49      | 0.16   | 0.24     |
| nv    | 0.77      | 0.96   | 0.85     |
| vasc  | 0.00      | 0.00   | 0.00     |

---

## Model Comparison

| Metric         | CNN       | Random Forest |
| -------------- | --------- | ------------- |
| Accuracy       | **73.1%** | 71.6%         |
| Macro F1-Score | **0.42**  | 0.29          |
| Weighted F1    | **0.71**  | 0.66          |

* CNN performed **better overall**, especially on common classes like `nv`.
* Both models struggled on minority classes such as `df` and `vasc`.
* CNN demonstrated better generalization and precision across the dataset, especially for imbalanced class distributions.

---

## Training Time

* **Random Forest**: Faster to train due to low model complexity and handcrafted features.
* **CNN**: Longer training time, but outperformed RF in overall classification accuracy and F1 scores.

---

## Conclusion

The CNN model outperforms Random Forest in terms of classification accuracy, especially for more frequent classes. While RF trains faster, it fails to generalize well across all classes, particularly in imbalanced data scenarios. Therefore:

* **CNN is more suitable** for real-world medical image classification tasks where accuracy is critical.
* **RF** can be useful for quick benchmarking or as part of an ensemble.

---

## Future Work

* Try **deeper CNN architectures** (e.g., ResNet, EfficientNet).
* Perform **data augmentation** to reduce overfitting and improve minority class detection.
* Combine models into an **ensemble** for better performance.
* Apply **transfer learning** to leverage pretrained models on medical images.

---

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/aAbaza3/skin-cancer-ml-vs-dl.git
   ```

2. Navigate to the project directory:

   ```bash
   cd skin-cancer-ml-vs-dl
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run training and evaluation:

   ```bash
   python main.py
   ```

5. View results:

   * Classification reports in terminal or log file
   * Performance saved in `comparison_results.csv`
   * Chart saved as `comparison_chart.png`

---

## Contributing

Pull requests and suggestions are welcome. Feel free to open issues or submit PRs.

---

--- 

This README structure presents the key components clearly and concisely, focusing on your latest results and comparison of the models.
