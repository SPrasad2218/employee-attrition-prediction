"""
Streamlit demo:  streamlit run app.py
Run attrition_project.py first (it creates outputs/attrition_model.joblib).
Only the most important inputs are shown; everything else uses the training-data median/mode.
"""
import joblib
import pandas as pd
import streamlit as st

bundle = joblib.load("outputs/attrition_model.joblib")
model, defaults, choices = bundle["model"], bundle["defaults"], bundle["choices"]

st.title("Employee Attrition Risk Predictor")
st.caption("Enter a few details to estimate the chance that an employee leaves.")

row = dict(defaults)  # start from typical values

c1, c2 = st.columns(2)
with c1:
    row["Age"] = st.slider("Age", 18, 60, int(defaults["Age"]))
    row["MonthlyIncome"] = st.number_input("Monthly income", 1000, 25000, int(defaults["MonthlyIncome"]), step=500)
    row["DistanceFromHome"] = st.slider("Distance from home (km)", 1, 30, int(defaults["DistanceFromHome"]))
    row["YearsAtCompany"] = st.slider("Years at company", 0, 40, int(defaults["YearsAtCompany"]))
with c2:
    row["OverTime"] = st.selectbox("Works overtime?", choices["OverTime"])
    row["JobSatisfaction"] = st.slider("Job satisfaction (1-4)", 1, 4, int(defaults["JobSatisfaction"]))
    row["WorkLifeBalance"] = st.slider("Work-life balance (1-4)", 1, 4, int(defaults["WorkLifeBalance"]))
    row["MaritalStatus"] = st.selectbox("Marital status", choices["MaritalStatus"])

if st.button("Predict"):
    X_new = pd.DataFrame([row])[bundle["columns"]]
    p = model.predict_proba(X_new)[0, 1]
    st.metric("Attrition probability", f"{p:.0%}")
    if p >= 0.5:
        st.error("High risk of leaving")
    elif p >= 0.3:
        st.warning("Moderate risk")
    else:
        st.success("Low risk")
    st.caption("Educational model - not for real HR decisions.")
