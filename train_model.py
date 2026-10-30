
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# 1. Load cleaned data
df = pd.read_csv("cleaned_orders.csv")

# 2. Prepare date and target
df["order_datetime"] = pd.to_datetime(
    df["order_datetime"]
)

df = df.sort_values("order_datetime").reset_index(drop=True)

df["is_rto"] = (
    df["final_status"] == "RTO"
).astype(int)

# 3. Extract useful date features
df["order_hour"] = df["order_datetime"].dt.hour
df["order_month"] = df["order_datetime"].dt.month

# 4. Check customer history consistency
invalid_history = (
    df["customer_prior_rto"] >
    df["customer_prior_orders"]
).sum()

print("Invalid customer history records:", invalid_history)

# 5. Select model features
numeric_features = [
    "order_value",
    "num_items",
    "customer_prior_orders",
    "customer_prior_rto",
    "order_hour",
    "order_month"
]

categorical_features = [
    "state",
    "courier",
    "utm_source"
]

features = numeric_features + categorical_features

X = df[features]
y = df["is_rto"]

# 6. Chronological train-test split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training RTO rate:", round(y_train.mean() * 100, 2), "%")
print("Testing RTO rate:", round(y_test.mean() * 100, 2), "%")

# 7. Preprocess numerical features
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 8. Preprocess categorical features
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 9. Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 10. Build Logistic Regression model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 11. Train model
model.fit(X_train, y_train)

# 12. Predict test probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Default threshold for initial evaluation
predictions = (probabilities >= 0.50).astype(int)

# 13. Baseline comparison
baseline = DummyClassifier(strategy="prior")
baseline.fit(X_train, y_train)

baseline_probabilities = baseline.predict_proba(X_test)[:, 1]
baseline_predictions = (baseline_probabilities >= 0.50).astype(int)

def evaluate(name, actual, predicted, probs):

    print(f"\n========== {name} ==========")

    print("Accuracy:", round(accuracy_score(actual, predicted), 4))
    print("Precision:", round(precision_score(actual, predicted, zero_division=0), 4))
    print("Recall:", round(recall_score(actual, predicted, zero_division=0), 4))
    print("F1-score:", round(f1_score(actual, predicted, zero_division=0), 4))
    print("ROC-AUC:", round(roc_auc_score(actual, probs), 4))
    print("PR-AUC:", round(average_precision_score(actual, probs), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(actual, predicted))

    print("\nClassification Report:")
    print(classification_report(actual, predicted, zero_division=0))

# 14. Evaluate both models
evaluate(
    "BASELINE MODEL",
    y_test,
    baseline_predictions,
    baseline_probabilities
)

evaluate(
    "LOGISTIC REGRESSION MODEL",
    y_test,
    predictions,
    probabilities
)

# 15. Save trained model
joblib.dump(model, "rto_model.joblib")

print("\nModel saved successfully as rto_model.joblib")

