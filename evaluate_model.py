
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# 1. Load data
df = pd.read_csv("cleaned_orders.csv")

df["order_datetime"] = pd.to_datetime(df["order_datetime"])
df = df.sort_values("order_datetime").reset_index(drop=True)

# 2. Target variable
y = (df["final_status"] == "RTO").astype(int)

# 3. Create date features
df["order_hour"] = df["order_datetime"].dt.hour
df["order_month"] = df["order_datetime"].dt.month

# 4. Select features
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

# 5. Chronological 60/20/20 split
n = len(df)

train_end = int(n * 0.60)
validation_end = int(n * 0.80)

X_train = X.iloc[:train_end]
X_validation = X.iloc[train_end:validation_end]
X_test = X.iloc[validation_end:]

y_train = y.iloc[:train_end]
y_validation = y.iloc[train_end:validation_end]
y_test = y.iloc[validation_end:]

print("Training records:", len(X_train))
print("Validation records:", len(X_validation))
print("Testing records:", len(X_test))

print("\nRTO rates:")
print("Training:", round(y_train.mean() * 100, 2), "%")
print("Validation:", round(y_validation.mean() * 100, 2), "%")
print("Testing:", round(y_test.mean() * 100, 2), "%")

# 6. Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 7. Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 8. Train ONLY on training data
model.fit(X_train, y_train)

# 9. Get validation probabilities
validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

# 10. Compare thresholds using validation data
print("\n========== VALIDATION THRESHOLD ANALYSIS ==========")

thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

for threshold in thresholds:

    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation, predictions, zero_division=0
    )

    recall = recall_score(
        y_validation, predictions, zero_division=0
    )

    f1 = f1_score(
        y_validation, predictions, zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

# 11. Select threshold using validation F1
# This is a provisional statistical threshold,
# not yet the final business-optimal threshold.

best_threshold = 0.50
best_f1 = -1

for threshold in np.arange(0.10, 0.61, 0.01):

    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_validation, predictions, zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)

print("\nSelected validation threshold:", round(best_threshold, 2))
print("Validation F1:", round(best_f1, 4))

# 12. Final test evaluation
test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\n========== FINAL TEST RESULTS ==========")

print("Threshold:", round(best_threshold, 2))

print(
    "Precision:",
    round(precision_score(y_test, test_predictions, zero_division=0), 4)
)

print(
    "Recall:",
    round(recall_score(y_test, test_predictions, zero_division=0), 4)
)

print(
    "F1:",
    round(f1_score(y_test, test_predictions, zero_division=0), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, test_probabilities), 4)
)

print(
    "PR-AUC:",
    round(average_precision_score(y_test, test_probabilities), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

# 13. Save model and threshold together
joblib.dump(
    {
        "model": model,
        "threshold": best_threshold,
        "features": features
    },
    "rto_model.joblib"
)

print("\nImproved model and threshold saved successfully!")
