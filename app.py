import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Load model
model = joblib.load("heart_disease_model.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("🫀 Heart Disease Prediction App")

st.markdown("""
This app predicts whether a patient is likely to have heart disease based on key health indicators.
You can also explore visual insights from the dataset.
""")

# Sidebar for input
st.sidebar.header("Patient Data Input")
def user_input_features():
    age = st.sidebar.slider("Age", 29, 77, 54)
    sex = st.sidebar.selectbox("Sex", [0, 1])
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure", 94, 200, 130)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 126, 564, 246)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG results", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 71, 202, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST depression", 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
    ca = st.sidebar.slider("# Major vessels (0-3) colored by flourosopy", 0, 3, 0)
    thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
st.subheader("Patient Input Features")
st.write(input_df)

# Predict
prediction = model.predict(input_df)[0]
pred_proba = model.predict_proba(input_df)[0][1]

st.subheader("Prediction")
st.success(f"Prediction: {'🔴 Heart Disease' if prediction else '🟢 No Heart Disease'}")
st.info(f"Probability of Heart Disease: {pred_proba:.2f}")

# SHAP for explainability
import shap
from shap import Explanation

# Compute SHAP values
explainer = shap.Explainer(model, input_df)
shap_values = explainer(input_df)

# Extract SHAP values for class 1 (heart disease)
shap_vals = shap_values.values[0][:, 1]  # class 1
base_val = shap_values.base_values[0][1]
data_row = input_df.iloc[0]

# Create single Explanation object
explanation = Explanation(
    values=shap_vals,
    base_values=base_val,
    data=data_row,
    feature_names=input_df.columns.tolist()
)

# Plot the waterfall chart
st.subheader("Feature Contribution (SHAP)")
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(explanation, show=False)
st.pyplot(fig)



# Load dataset for visual analysis (you can customize or replace this)
df = pd.read_csv("heart.csv")

st.subheader("📊 Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x="age", color="target", barmode="overlay",
                        labels={"target": "Heart Disease"}, title="Age Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.box(df, x="target", y="chol", color="target",
                  labels={"target": "Heart Disease", "chol": "Cholesterol"},
                  title="Cholesterol vs Heart Disease")
    st.plotly_chart(fig2, use_container_width=True)

# Correlation heatmap
st.subheader("Correlation Heatmap")
corr = df.corr()
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(fig3)
