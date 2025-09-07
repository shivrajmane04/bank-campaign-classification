# streamlit_app.py
import streamlit as st
import requests
import pandas as pd

API_URL = API_URL = "http://localhost:5000/predict"
  # or set in Streamlit secrets

st.set_page_config(page_title="Bank Marketing Predictor", layout="centered")

st.title("Bank Marketing Campaign — Predict Term Deposit Subscription")

st.markdown("Fill client information and click **Predict** (calls Flask backend).")

with st.form(key='predict_form'):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=16, max_value=120, value=40)
        job = st.selectbox("Job", ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"])
        marital = st.selectbox("Marital", ["married","divorced","single","unknown"])
        education = st.selectbox("Education", ["unknown","secondary","primary","tertiary"])
        default = st.selectbox("Default (credit)", ["no","unknown","yes"])
        balance = st.number_input("Balance", value=1000)
        housing = st.selectbox("Housing loan", ["yes","no","unknown"])
        loan = st.selectbox("Personal loan", ["yes","no","unknown"])
    with col2:
        contact = st.selectbox("Contact", ["unknown","telephone","cellular"])
        day = st.number_input("Last contact day of month", min_value=1, max_value=31, value=15)
        month = st.selectbox("Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
        duration = st.number_input("Last contact duration (s)", min_value=0, value=200)
        campaign = st.number_input("Campaign (contacts in this campaign)", min_value=1, value=1)
        pdays = st.number_input("Days since last contact (-1 if never)", value=-1)
        previous = st.number_input("Number of contacts previously", min_value=0, value=0)
        poutcome = st.selectbox("Outcome of previous campaign", ["unknown","other","failure","success"])

    submit = st.form_submit_button("Predict")

if submit:
    payload = {
        "age": int(age),
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": float(balance),
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": int(day),
        "month": month,
        "duration": int(duration),
        "campaign": int(campaign),
        "pdays": int(pdays),
        "previous": int(previous),
        "poutcome": poutcome
    }

    try:
        with st.spinner("Calling backend..."):
            r = requests.post(API_URL, json=payload, timeout=10)
        if r.status_code == 200:
            res = r.json()
            pred = res.get("prediction", "N/A")
            prob = res.get("probability", None)
            st.success(f"Prediction: **{pred.upper()}**")
            if prob is not None:
                st.info(f"Probability of subscription: **{prob:.4f}**")
        else:
            st.error(f"API error: {r.status_code} - {r.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")


