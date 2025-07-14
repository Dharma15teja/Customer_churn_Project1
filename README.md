# 🧠 Customer Churn Prediction (Telco Dataset)

This project builds a machine learning model to predict whether a customer is likely to churn (leave the service) using telecom customer data. It uses a Random Forest Classifier with a full preprocessing pipeline and hyperparameter tuning to achieve strong predictive performance.

---

## 📌 Project Goals

- Predict customer churn based on historical telecom usage data.
- Handle both categorical and numerical features using a clean preprocessing pipeline.
- Use `GridSearchCV` to tune hyperparameters of a Random Forest model.
- Evaluate the model using cross-validation and classification metrics.

---

## 📊 Dataset

- Source: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- Size: ~7,000 rows × 20 columns
- Features include:
  - Customer demographics
  - Account information (tenure, contract type, payment method)
  - Service usage patterns
  - Target: `Churn` (Yes/No)

---

## 🔧 Technologies Used

- Python (Pandas)
- Scikit-learn (RandomForest, GridSearchCV, Pipelines)
- Colab

---

## ✅ Model Pipeline

1. **Data Cleaning**
   - Convert `TotalCharges` to numeric
   - Handle missing values

2. **Preprocessing**
   - Encode categorical features using `OrdinalEncoder`
   - Scale numerical features with `StandardScaler`
   - Combined using `ColumnTransformer` inside a `Pipeline`

3. **Model**
   - `RandomForestClassifier`
   - Tuned using `GridSearchCV` over 5-fold CV

---

## 🏆 Best Model Results

- **Model**: Random Forest Classifier
- **Best Hyperparameters**:
  ```python
  {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
  
  }
