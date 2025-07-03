#  Heart Disease Prediction using Machine Learning

## Overview

This project builds a machine learning model to predict the presence of heart disease based on clinical indicators such as age, sex, chest pain type, cholesterol, etc. It includes complete data analysis, model training, evaluation, and deployment via a Streamlit web app.

---

##  Key Links

- 🌐 **Web App**: [https://heart-disease-prediction-vb9zqtvqb2soyrfgqguwpp.streamlit.app/)
- 📑 **Analysis Report (PDF)**: [https://github.com/garvitkumbhat619/Heart-Disease-prediction/blob/047d0268afe0e26f185c157c700ebabd1dd6804a/heart-disease-prdiction-analysis.pdf)

---

## Objective

To accurately classify whether an individual has heart disease using patient data and machine learning models, with explainable predictions and a user-friendly deployment interface.

---

## Dataset Description

- Source: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- Total Records: 1025
- No missing values

| Feature     | Description                                  |
|-------------|----------------------------------------------|
| age         | Age in years                                 |
| sex         | 1 = Male, 0 = Female                         |
| cp          | Chest pain type (0–3)                        |
| trestbps    | Resting blood pressure                       |
| chol        | Serum cholesterol (mg/dl)                    |
| fbs         | Fasting blood sugar > 120 mg/dl              |
| restecg     | Resting ECG results                          |
| thalach     | Max heart rate achieved                      |
| exang       | Exercise-induced angina                      |
| oldpeak     | ST depression during exercise                |
| slope       | Slope of the peak ST segment                 |
| ca          | Major vessels colored by fluoroscopy         |
| thal        | Thalassemia result                           |
| target      | 0 = No disease, 1 = Has disease              |

---

##  Exploratory Data Analysis

###  Heart Disease Distribution by Sex

![Sex vs Heart Disease](https://github.com/garvitkumbhat619/Heart-Disease-prediction/blob/e898e9521d6053409540a5ab420ad6e2042d81fe/sex%20vs%20heart_disease_count.png)

- Females (sex = 0): ~75% had heart disease  
- Males (sex = 1): ~45% had heart disease

---

### Age vs Max Heart Rate by Disease

![Age vs Thalach](https://github.com/garvitkumbhat619/Heart-Disease-prediction/blob/e898e9521d6053409540a5ab420ad6e2042d81fe/age%20vs%20max_heart_rate.png)

- Younger individuals with high heart rates are more prone to heart disease.
- In older ages, heart disease appears across a broader heart rate range.

---

##  Model Performance

We compared several models:

| Model                         | Accuracy | Precision | Recall | F1 Score |
|------------------------------|----------|-----------|--------|----------|
| Logistic Regression (base)   | 79.5%    | 75.6%     | 87.4%  | 81.1%    |
| Random Forest (base)         | **98.5%**| **100%**  | 97.1%  | **98.5%**|
| Logistic Regression (tuned)  | 78.5%    | 74.3%     | 87.4%  | 80.4%    |
| Random Forest (tuned)        | 95.6%    | 95.2%     | 96.1%  | 95.6%    |

---

## Evaluation Metrics

### Cross-Validated Metrics (Logistic Regression)

![Cross-Validation Metrics](https://github.com/garvitkumbhat619/Heart-Disease-prediction/blob/e898e9521d6053409540a5ab420ad6e2042d81fe/Cross-Validated%20Metrics.png)

- Accuracy: 84.8%
- Precision: 82.1%
- Recall: 90.0%
- F1 Score: 86.0%
- ROC AUC: 92.0%

---

##  Feature Importance

![Feature Importance](https://github.com/garvitkumbhat619/Heart-Disease-prediction/blob/e898e9521d6053409540a5ab420ad6e2042d81fe/feature_imp.png)

- `sex`, `thal`, and `ca` were the most influential features.
- Positive coefficients (e.g., `cp`, `slope`) increase likelihood of disease.
- Negative coefficients (e.g., `sex`, `thal`) reduce the likelihood.

---
## Future Improvements

- Add SVM, XGBoost models
- Expand hyperparameter tuning
- Deploy with Docker or cloud platforms (Heroku, AWS)
- Improve UI and collect more real-world samples

---


