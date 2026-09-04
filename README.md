# 🩺 Breast Cancer Diagnosis Classification using KNN

An end-to-end Machine Learning project to classify breast tumors as either **Benign** or **Malignant** using the **Breast Cancer Wisconsin (Diagnostic)** dataset and an optimized **K-Nearest Neighbors (KNN)** classifier.

---

## 📌 Project Overview
Early detection of breast cancer significantly increases the chances of successful treatment. This project implements a classification pipeline with a strong emphasis on medical evaluation metrics—specifically minimizing **False Negatives** (failing to detect a malignant tumor) by optimizing the model's **Recall**.

---

## 🎯 Key Highlights
* **High Accuracy:** Achieved **99.12%** overall accuracy on unseen test data.
* **Clinical Safety (Recall Focus):** Reached a **0.98 Recall** on malignant cases (detected 42 out of 43 malignant tumors, with only 1 false negative).
* **Targeted Feature Selection:** Reduced the feature space from 30 features down to 7 core high-impact geometric indicators.
* **Proper Preprocessing:** Implemented feature standardization via `StandardScaler` to prevent feature dominance in Euclidean distance calculations.

---

## 🔬 Selected Features
Instead of using all 30 features, the following 7 features with the strongest correlation to tumor malignancy were selected:
* `radius_mean`
* `perimeter_mean`
* `area_mean`
* `concave points_mean`
* `radius_worst`
* `perimeter_worst`
* `area_worst`

---

## 📊 Evaluation & Results

### Confusion Matrix
```text
[[71   0]
 [ 1  42]]
