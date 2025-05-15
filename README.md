
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

---

## Results

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

