# 🏦 Credit Risk ML System

An end-to-end **Machine Learning powered credit risk prediction system** that predicts whether a loan applicant is **Safe or High Risk**.

This project combines **Machine Learning, API development, and an interactive dashboard** to simulate how real fintech companies evaluate loan applicants.

Users can enter customer financial details and instantly receive:

✔ Risk prediction
✔ Probability score
✔ Model explanation
✔ Feature impact visualization

---

🧠 Problem

Banks and financial institutions must evaluate whether a borrower is likely to **default on a loan**.

Traditional risk assessment is slow and subjective.
This project demonstrates how **Machine Learning can automate credit risk evaluation** while also providing **model explainability**.

---

⚙️ System Architecture

User Input
↓
Streamlit Dashboard
↓
FastAPI Backend
↓
Random Forest ML Model
↓
Prediction + Explainability

---

✨ Features

Real-time credit risk prediction
Interactive ML dashboard
Explainable AI with feature impact visualization
Fast API-based inference pipeline
Model performance analytics

---

🧪 Model Performance

| Metric   | Score         |
| -------- | ------------- |
| Accuracy | **78%**       |
| ROC-AUC  | **0.80**      |
| Model    | Random Forest |

The model predicts whether a loan applicant is **high risk or safe** based on financial attributes.

---

🛠 Tech Stack

### 👨‍💻 Language

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)

### 🤖 Machine Learning

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)

### ⚡ Backend

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge\&logo=fastapi\&logoColor=white)

### 📊 Dashboard

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)

### 📈 Visualization

![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge\&logo=plotly\&logoColor=white)

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)

![Seaborn](https://img.shields.io/badge/Seaborn-5A9?style=for-the-badge)

---

📂 Project Structure

credit-risk-ml-system

app
├── main.py
├── model
│   ├── model.pkl
│   └── predict.py

data
└── german_credit.csv

ui.py
requirements.txt
README.md

---

🚀 Running the Project Locally

1️⃣ Clone the repository

git clone https://github.com/Hardikshah126/credit-risk-ml-system.git

cd credit-risk-ml-system

---

2️⃣ Create virtual environment

python -m venv venv

venv\Scripts\activate

---

3️⃣ Install dependencies

pip install -r requirements.txt

---

4️⃣ Start FastAPI backend

uvicorn app.main:app --reload

---

5️⃣ Launch Streamlit dashboard

streamlit run ui.py

---

Open the dashboard in your browser:

http://localhost:8501

---

📊 Dashboard Overview

### Prediction Dashboard

Users can enter loan applicant details and receive a **risk score and prediction**.

### Model Explanation

The system visualizes **feature impact** to explain why the model predicted a certain risk level.

### Model Performance

Displays:

• Accuracy
• ROC Curve
• Feature Importance

---

# 🎯 Key Learnings

While building this project I explored:

• Building end-to-end ML systems
• Deploying models with APIs
• Creating interactive ML dashboards
• Implementing explainable AI for model transparency

---

# 🔮 Future Improvements

📈 Hyperparameter tuning to improve accuracy
🌍 Cloud deployment for public access
🧠 Advanced models like XGBoost or LightGBM
📊 More detailed explainability visualizations

---

👨‍💻 Author

**Hardik Shah**

AI • Machine Learning • Design

GitHub
https://github.com/Hardikshah126

---

⭐ If you found this project interesting, consider giving it a star!
