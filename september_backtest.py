
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ==================================================
# 1. LOAD DATA AND CONFIGURATION
# ==================================================

data = pd.read_csv("cleaned_orders.csv")

with open("business_config.json", "r") as file:
    config = json.load(file)

data["order_datetime"] = pd.to_datetime(
    data["order_datetime"], errors="coerce"
)

data["final_status"] = (
    data["final_status"].astype(str).str.strip().str.upper()
)

data = data.drop_duplicates()

data = data[
    data["final_status"].isin(["DELIVERED", "RTO"])
].copy()

data = data.dropna(subset=["order_datetime"])

data = data.sort_values("order_datetime").reset_index(drop=True)


# ==================================================
# 2. CREATE DATE-BASED TRAIN AND TEST DATA
# ==================================================

# Train: April to August 2026
# Test: September 2026

train_data = data[
    data["order_datetime"] < "2026-09-01"
].copy()

test_data = data[
    (data["order_datetime"] >= "2026-09-01")
    & (data["order_datetime"] < "2026-10-01")
].copy()

if train_data.empty or test_data.empty:
    raise ValueError(
        "Training or September test data is empty. Check the dataset dates."
    )

print("\n========== DATA SPLIT ==========")
print("Training orders:", len(train_data))
print("September test orders:", len(test_data))

print(
    "Training period:",
    train_data["order_datetime"].min(),
    "to",
    train_data["order_datetime"].max()
)

print(
    "Test period:",
    test_data["order_datetime"].min(),
    "to",
    test_data["order_datetime"].max()
)


# ==================================================
# 3. FEATURE ENGINEERING
# ==================================================

for frame in [train_data, test_data]:

    frame["order_hour"] = frame["order_datetime"].dt.hour
    frame["order_month"] = frame["order_datetime"].dt.month

features = [
    "order_value",
    "num_items",
    "customer_prior_orders",
    "customer_prior_rto",
    "order_hour",
    "order_month",
    "state",
    "courier",
    "utm_source"
]

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

X_train = train_data[features]
X_test = test_data[features]

y_train = (
    train_data["final_status"] == "RTO"
).astype(int)

y_test = (
    test_data["final_status"] == "RTO"
).astype(int)


# ==================================================
# 4. BUILD MODEL PIPELINE
# ==================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)


# ==================================================
# 5. TRAIN AND PREDICT
# ==================================================

model.fit(X_train, y_train)

test_data["rto_risk"] = model.predict_proba(X_test)[:, 1]

test_data["risk_percentage"] = (
    test_data["rto_risk"] * 100
).round(2)


# ==================================================
# 6. MODEL PERFORMANCE
# ==================================================


# ==================================================
# 6. MODEL PERFORMANCE
# ==================================================

# Compare the standard threshold with the threshold
# selected earlier using validation data.
# September remains an untouched test period.

print("\n========== SEPTEMBER MODEL PERFORMANCE ==========")

for threshold in [0.50, 0.18]:

    predictions = (
        test_data["rto_risk"] >= threshold
    ).astype(int)

    print(f"\nClassification threshold: {threshold}")

    print("Precision:", round(
        precision_score(
            y_test, predictions, zero_division=0
        ), 4
    ))

    print("Recall:", round(
        recall_score(
            y_test, predictions, zero_division=0
        ), 4
    ))

    print("F1 Score:", round(
        f1_score(
            y_test, predictions, zero_division=0
        ), 4
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# Ranking metrics do not depend on the classification threshold.
print("\nRanking performance:")

print("ROC-AUC:", round(
    roc_auc_score(y_test, test_data["rto_risk"]), 4
))

print("PR-AUC:", round(
    average_precision_score(y_test, test_data["rto_risk"]), 4
))



# ==================================================
# 7. APPLY BUSINESS DECISION RULES
# ==================================================

thresholds = config["risk_thresholds"]


def get_decision(risk):

    if risk < thresholds["ship_as_is"]:
        return "Ship as-is"

    elif risk < thresholds["confirmation"]:
        return "WhatsApp confirmation"

    elif risk < thresholds["prepaid_nudge"]:
        return "Prepaid nudge"

    else:
        return "Hold for manual review"


test_data["decision"] = (
    test_data["rto_risk"].apply(get_decision)
)


# ==================================================
# 8. SUMMARISE DECISIONS AND ACTUAL OUTCOMES
# ==================================================

test_data["is_rto"] = (
    test_data["final_status"] == "RTO"
).astype(int)

summary = test_data.groupby(
    "decision"
).agg(
    orders=("order_id", "count"),
    actual_rto=("is_rto", "sum"),
    delivered=("is_rto", lambda x: (x == 0).sum()),
    average_risk=("risk_percentage", "mean")
).reindex(
    [
        "Ship as-is",
        "WhatsApp confirmation",
        "Prepaid nudge",
        "Hold for manual review"
]
).fillna(0)

summary["actual_rto_rate"] = (
    summary["actual_rto"] / summary["orders"].replace(0, pd.NA) * 100
)

print("\n========== SEPTEMBER DECISION SUMMARY ==========")
print(summary.round(2).to_string())


# ==================================================
# 9. SCENARIO-BASED FINANCIAL ESTIMATE
# ==================================================

rto_cost = float(config["rto_cost"])
discount = float(config["prepaid_discount"])

# These assumptions are illustrative, not measured results.
intervention_effectiveness = 0.50
prepaid_conversion_rate = 0.20

intervention_orders = test_data[
    test_data["decision"].isin(
        ["WhatsApp confirmation", "Prepaid nudge"]
    )
]

estimated_rto_savings = (
    intervention_orders["rto_risk"].sum()
    * rto_cost
    * intervention_effectiveness
)

prepaid_orders = (
    test_data["decision"] == "Prepaid nudge"
).sum()

expected_prepaid_conversions = (
    prepaid_orders * prepaid_conversion_rate
)

discount_cost = (
    expected_prepaid_conversions * discount
)

# No automatic cancellations in this policy.
cancellation_loss = 0

net_impact = (
    estimated_rto_savings
    - discount_cost
    - cancellation_loss
)

print("\n========== SEPTEMBER SCENARIO ESTIMATE ==========")
print(f"Estimated RTO savings: Rs. {estimated_rto_savings:.2f}")
print(f"Expected discount cost: Rs. {discount_cost:.2f}")
print(f"Genuine-order cancellation loss: Rs. {cancellation_loss:.2f}")
print(f"Estimated net impact: Rs. {net_impact:.2f}")

print(
    "\nNote: These are scenario estimates, not realised savings."
)


# ==================================================
# 10. SAVE SEPTEMBER DECISION LOG
# ==================================================

display_columns = [
    "order_id",
    "order_datetime",
    "customer_name",
    "state",
    "order_value",
    "rto_risk",
    "risk_percentage",
    "decision",
    "final_status"
]

decision_log = test_data[display_columns].copy()

decision_log = decision_log.sort_values(
    "rto_risk",
    ascending=False
)

decision_log.to_csv(
    "september_backtest.csv",
    index=False
)

summary.to_csv(
    "september_decision_summary.csv"
)

print("\nFiles saved successfully:")
print("1. september_backtest.csv")
print("2. september_decision_summary.csv")
