import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Credit Risk AI", layout="wide")

st.title("🏦 Credit Risk AI Dashboard")
st.write("Predict loan default probability using machine learning.")

# ---------- Tabs ----------
tabs = st.tabs(["Prediction", "Model Performance"])

# =========================================================
# TAB 1 — PREDICTION
# =========================================================
with tabs[0]:

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 75, 30)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.selectbox("Job Level", [0, 1, 2, 3])
        housing = st.selectbox("Housing", ["own", "rent", "free"])

        saving_accounts = st.selectbox(
            "Saving Accounts",
            ["little", "moderate", "quite rich", "rich"]
        )

    with col2:
        checking_account = st.selectbox(
            "Checking Account",
            ["little", "moderate", "rich"]
        )

        credit_amount = st.slider(
            "Credit Amount",
            100,
            20000,
            5000
        )

        duration = st.slider(
            "Loan Duration (months)",
            6,
            60,
            12
        )

        purpose = st.selectbox(
            "Loan Purpose",
            ["car", "radio/TV", "education", "furniture/equipment", "business"]
        )

    if st.button("Predict Risk"):

        data = {
            "Age": age,
            "Sex": sex,
            "Job": job,
            "Housing": housing,
            "Saving_accounts": saving_accounts,
            "Checking_account": checking_account,
            "Credit_amount": credit_amount,
            "Duration": duration,
            "Purpose": purpose
        }

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        if response.status_code != 200:
            st.error("FastAPI server error")
            st.stop()

        result = response.json()

        prediction = result["prediction"]
        prob = result["risk_probability"]
        explanations = result["explanations"]

        st.subheader("Prediction Result")

        colA, colB = st.columns(2)

        with colA:
            if prediction == 1:
                st.error("⚠ HIGH RISK CUSTOMER")
            else:
                st.success("✅ SAFE CUSTOMER")

        with colB:
            st.metric(
                "Default Probability",
                f"{prob*100:.1f}%"
            )

        # ---------- Gauge ----------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': 'green'},
                    {'range': [40, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'red'}
                ]
            }
        ))

        st.plotly_chart(fig, width="stretch")

        # ---------- SHAP Explanation ----------
        st.subheader("Why the model predicted this")

        df = pd.DataFrame(explanations)
        df["impact"] = df["impact"].round(3)

        st.bar_chart(df.set_index("feature"))

# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================
with tabs[1]:

    st.header("📊 Model Performance Dashboard")

    # Load trained model
    model = joblib.load("app/model/model.pkl")

    # ---------- Metrics ----------
    st.subheader("Model Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", "78%")

    with col2:
        st.metric("ROC AUC", "0.80")

    st.info(
        "Metrics were calculated during model training on the German Credit dataset."
    )

    # ---------- Feature Importance ----------
    st.subheader("Feature Importance")

    feature_names = [
        "Age",
        "Sex",
        "Job",
        "Housing",
        "Saving Accounts",
        "Checking Account",
        "Credit Amount",
        "Duration",
        "Purpose"
    ]

    importances = model.feature_importances_

    feature_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    feature_df = feature_df.sort_values(
        by="importance",
        ascending=False
    )

    st.bar_chart(feature_df.set_index("feature"))