import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/german_credit.csv")

# Drop unnecessary column
df = df.drop(columns=["Unnamed: 0"])

# Convert target
df["Risk"] = df["Risk"].map({"good": 0, "bad": 1})

# Separate features & target
X = df.drop("Risk", axis=1)
y = df["Risk"]

# Fill missing categorical values
X["Saving accounts"] = X["Saving accounts"].fillna("unknown")
X["Checking account"] = X["Checking account"].fillna("unknown")

# Encode categorical columns
label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.4).astype(int)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Save artifacts
joblib.dump(model, "app/model/model.pkl")
joblib.dump(scaler, "app/model/scaler.pkl")
joblib.dump(label_encoders, "app/model/encoders.pkl")

print("\nModel saved successfully!")


# Feature importance from Random Forest
importances = model.feature_importances_

# Get feature names
feature_names = X.columns

# Create DataFrame
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)

# Plot
plt.figure()
plt.bar(feature_importance_df["Feature"], feature_importance_df["Importance"])
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()