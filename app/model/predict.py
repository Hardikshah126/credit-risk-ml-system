import joblib
import numpy as np
import shap

# Load saved model artifacts
model = joblib.load("app/model/model.pkl")
scaler = joblib.load("app/model/scaler.pkl")
encoders = joblib.load("app/model/encoders.pkl")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)


def predict_credit_risk(data: dict):

    column_map = {
        "Saving_accounts": "Saving accounts",
        "Checking_account": "Checking account",
        "Credit_amount": "Credit amount"
    }

    features = []
    feature_names = []

    # ---------- Encode features ----------
    for key, value in data.items():

        col_name = column_map.get(key, key)

        if col_name in encoders:
            value = encoders[col_name].transform([value])[0]

        features.append(value)
        feature_names.append(col_name)

    # Convert to numpy
    features = np.array(features).reshape(1, -1)

    # ---------- Scale ----------
    scaled = scaler.transform(features)

    # ---------- Model Prediction ----------
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    # ---------- SHAP Explanation ----------
    shap_values = explainer.shap_values(scaled)

    # Handle different SHAP formats
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    shap_list = []

    for name, value in zip(feature_names, shap_vals):

        # Convert SHAP array safely to float
        impact = float(np.array(value).reshape(-1)[0])

        shap_list.append({
            "feature": name,
            "impact": impact
        })

    # Sort by strongest impact
    shap_list = sorted(
        shap_list,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return {
        "prediction": int(prediction),
        "risk_probability": float(probability),
        "explanations": shap_list[:5]
    }