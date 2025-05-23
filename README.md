
# ⚡ Electricity Theft Detection in Power Grids using CNN and Random Forests

This project presents a hybrid machine learning model combining **Convolutional Neural Networks (CNN)** and **Random Forests (RF)** to detect electricity theft in power grids using smart meter consumption data. The model is designed to enhance the accuracy and efficiency of electricity theft detection compared to traditional methods.

---

## 🧠 Project Overview

Electricity theft is a major cause of non-technical losses in power grids, leading to billions in losses and increased safety risks. This project leverages **deep learning** for feature extraction and **ensemble learning** for classification to identify abnormal consumption patterns that indicate potential theft.

---

## 🔍 Features

* Upload and preprocess real-world smart meter datasets
* Train and evaluate multiple models: CNN, Random Forest, SVM, CNN+RF, CNN+SVM
* Display precision, recall, F-score, and accuracy for each model
* Predict electricity theft in unseen test data
* Visual comparison of model performance via bar charts

---

## 🛠 Modules

* **Data Preprocessing**: Cleans missing and non-numeric values, normalizes the data.
* **CNN Model Training**: Extracts deep features from time-series consumption data.
* **Random Forest & SVM Training**: Trains on CNN-extracted or raw features.
* **Model Evaluation**: Calculates performance metrics for each algorithm.
* **Theft Prediction**: Predicts whether a given record indicates theft.
* **Comparison Graphs**: Visualizes and compares accuracy across algorithms.

---

## 📊 Technologies Used

* **Language**: Python 3.7
* **Libraries**:

  * TensorFlow – deep learning
  * Scikit-learn – machine learning models and metrics
  * NumPy, Pandas – data handling
  * Matplotlib – data visualization

---

## 📂 Dataset

* Uses real electricity consumption data from utility consumers.
* Labels indicate theft (`1`) or normal usage (`0`).
* Test data does not contain labels and is used for model prediction.

---

## 🖥 System Requirements

**Hardware:**

* RAM: 512 MB+
* Processor: Pentium IV 2.4 GHz or above

**Software:**

* OS: Windows
* Python 3.7
* Required packages (install via pip): `tensorflow`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`

---

## 🚀 How to Run

1. Clone the repository

   ```bash
   git clone <your-repo-url>
   cd electricity-theft-detection
   ```
2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```
3. Run the application

   ```bash
   python run.py
   ```

Alternatively, use `run.bat` for GUI-based interaction.

---

## 🧪 Results

* **CNN-RF Accuracy**: 100%
* **CNN-SVM Accuracy**: 99%
* **RF (only)**: 94%
* **SVM (only)**: 96%

The **CNN-RF model** outperforms all others by automatically extracting features and classifying with high accuracy.

---

## 🔒 Security & Ethics

* **Data Privacy**: User-level access controls using Row-Level Security (RLS).
* **Ethical Use**: Designed for authorized utility usage only; detection results should be verified manually before action.

---

## 📈 Future Enhancements

* Incorporate **differential privacy** to protect consumer data.
* Extend the hybrid model to related use-cases like **load forecasting**.
* Enable **real-time streaming detection** for continuous monitoring.

---

## 📚 References

Based on academic work by:

> *"Electricity Theft Detection in Power Grids with Deep Learning and Random Forests"*

Key sources include:

* SEAI and LCL smart meter datasets
* TensorFlow and Scikit-learn libraries
* Research on deep learning feature extraction and ensemble methods

