# 🧠 Multi-Class Disease Prediction

This project aims to predict disease categories using ensemble machine learning models while handling imbalanced medical data.

## 🔧 Tools & Tech
- Python, scikit-learn, XGBoost, imbalanced-learn, Pandas, NumPy
- Optional: TensorFlow, Keras, PyTorch, MLflow

## 🚀 Project Structure
```
multi-class-disease-prediction/
├── data/
│   └── dataset.csv
├── notebooks/
│   └── model_comparison.ipynb
├── scripts/
│   └── train_model.py
├── requirements.txt
└── README.md
```

## 🚀 How to Run
1. Clone the repo
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place your processed numpy arrays in the root or adapt paths in scripts
4. Run training:
   ```
   python scripts/train_model.py
   ```

## 📊 Results
- Classification reports
- ROC-AUC scores
- Confusion matrix
