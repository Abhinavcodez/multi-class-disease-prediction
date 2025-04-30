# 🧠 Multi-Class Disease Prediction

This machine learning project focuses on predicting disease categories (Diabetes, Heart Disease, Cancer, Healthy) from patient medical records using ensemble learning techniques. The dataset was synthetically generated for testing and includes key health metrics like age, BMI, blood pressure, and glucose levels.

---

## 📁 Dataset

The dataset (`dataset.csv`) includes 600 samples with the following features:

- `age`: Patient's age  
- `bmi`: Body Mass Index  
- `blood_pressure`: Systolic blood pressure  
- `cholesterol_level`: Cholesterol level (mg/dL)  
- `glucose`: Glucose level (mg/dL)  
- `gender`: 'M' or 'F'  
- `disease`: Target class (one of 'Healthy', 'Diabetes', 'Heart Disease', 'Cancer')

Class distribution is imbalanced to reflect real-world medical data.

---

## 🧪 Problem Statement

The goal is to classify patients into one of multiple disease categories based on their health metrics. Special care is taken to handle class imbalance using oversampling techniques like SMOTE.

---

## 🛠️ Approach

### 📌 Preprocessing

- Label Encoding for `gender`  
- Standardization of numerical features  
- SMOTE oversampling for balancing the training data  

### 🤖 Models Used

- Random Forest  
- XGBoost  
- AdaBoost  
- Logistic Regression  

All models were evaluated using stratified splits and classification reports. The `Random Forest` and `XGBoost` classifiers gave the best balance of precision and recall.

---

## 📈 Results

- **Evaluation Metrics**: Precision, Recall, F1-score for each class  
- **Observations**:
  - Random Forest performed best overall  
  - SMOTE helped significantly with minority class recall  
  - Logistic Regression struggled with non-linear patterns  

---

## 🤔 Learnings

- Handling imbalanced datasets is crucial in medical ML tasks  
- Ensemble models like Random Forest and XGBoost are robust across class boundaries  
- Proper preprocessing and feature scaling can greatly improve baseline performance  

---

## 🚀 Future Work

- Use real-world datasets (e.g. from UCI or Kaggle)  
- Explore deep learning methods with Keras or PyTorch  
- Incorporate feature importance visualizations  
- Apply SHAP or LIME for model explainability  

---

## 🗣️ Feedback Welcome!

Feel free to fork the repo, suggest improvements, or open issues for bugs or enhancements. Feedback is always appreciated!

[👉 View the Notebook](notebooks/model_comparison.ipynb)
