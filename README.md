# 🏦 Loan Approval Predictor

A machine learning project that predicts whether a loan application will be **approved or rejected** based on applicant details such as income, credit history, education, and property area. Built using Python and scikit-learn.

## 📌 Project Overview

This project uses historical loan application data to train and compare multiple classification models, then deploys the best-performing model to predict loan approval outcomes for new applicants.

## 🗂️ Dataset

The dataset (`loan_data.csv`) contains **614 records** with the following features:

| Column | Description |
|---|---|
| `Loan_ID` | Unique loan application ID |
| `Gender` | Applicant gender |
| `Married` | Marital status |
| `Dependents` | Number of dependents |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Self-employment status |
| `ApplicantIncome` | Applicant's income |
| `CoapplicantIncome` | Co-applicant's income |
| `LoanAmount` | Loan amount requested |
| `Loan_Amount_Term` | Term of the loan (in days) |
| `Credit_History` | Credit history meets guidelines (1) or not (0) |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | Target variable — Approved (Y) / Rejected (N) |

## 🛠️ Tools & Libraries

- **Python**
- **pandas**, **numpy** — data manipulation
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — modeling & evaluation
- **joblib** — model serialization

## 🔍 Workflow

### 1. Data Exploration & Cleaning
- Inspected dataset shape, structure, and missing values
- Filled missing categorical values with **mode**
- Filled missing numerical values (`LoanAmount`, `Loan_Amount_Term`) with **median/mode**

### 2. Feature Engineering
- Created `Total_Income` = `ApplicantIncome` + `CoapplicantIncome`
- Applied **log transformation** to `Total_Income` and `LoanAmount` to reduce skewness
- Converted `Dependents` value `'3+'` to numeric `3`
- Label-encoded categorical columns: `Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area`, `Loan_Status`
- Dropped `Loan_ID` (non-predictive identifier)

### 3. Model Training & Comparison
Split data into train (80%) / test (20%) sets and trained three classifiers:

| Model | Accuracy |
|---|---|
| Logistic Regression | **78.86%** |
| Decision Tree | 69.11% |
| Random Forest | 77.24% |

### 4. Model Evaluation
- Generated confusion matrix and classification report for Logistic Regression
- Re-trained on **standardized (scaled)** features using `StandardScaler`
- Evaluated feature importance using a Random Forest model — **Credit History** emerged as the most influential feature, followed by income-related features

### 5. Model Deployment
- Saved the final Logistic Regression model (`loan_model.pkl`) and the fitted scaler (`scaler.pkl`) using `joblib`
- Tested the pipeline on a sample applicant to verify prediction and probability output

## 📁 File Structure

```
├── loan_project.ipynb    # Full analysis & model training notebook
├── loan_data.csv          # Dataset
├── loan_model.pkl          # Saved trained model
├── scaler.pkl               # Saved StandardScaler
├── app.py                   # Application script
├── requirements.txt         # Project dependencies
└── README.md                 # Project documentation
```

## 🚀 How to Run

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run the notebook:
   ```bash
   jupyter notebook loan_project.ipynb
   ```
3. To use the saved model for predictions:
   ```python
   import joblib
   import pandas as pd

   model = joblib.load("loan_model.pkl")
   scaler = joblib.load("scaler.pkl")

   # Prepare applicant data in the same feature order as training
   sample_scaled = scaler.transform(sample_df)
   prediction = model.predict(sample_scaled)
   ```

## 📊 Key Insights

- **Credit History** is by far the strongest predictor of loan approval.
- Income-related features (Applicant Income, Total Income, Loan Amount) also play a significant role.
- Demographic features like Gender and Self-Employed status have relatively low predictive importance.

## 👤 Authors

- **Sameep Madan**
- **Somya Gupta**
