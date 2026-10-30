
import joblib
import pandas as pd
import json

# Load trained model and selected threshold
bundle = joblib.load("rto_model.joblib")
model = bundle["model"]

# Load business rules
with open("business_config.json", "r") as file:
    config = json.load(file)


def predict_order_risk(order):
    """
    Predict the estimated RTO risk for one COD order.
    """

    # Convert order details into a DataFrame
    order_df = pd.DataFrame([order])

    # Use exactly the features used during model training
    features = bundle["features"]
    order_df = order_df[features]

    # Get estimated RTO risk
    probability = model.predict_proba(order_df)[0][1]

    return probability


def decide_action(order):
    risk = predict_order_risk(order)

    thresholds = config["risk_thresholds"]

    if risk < thresholds["ship_as_is"]:
        action = "Ship as-is"
        reason = "Low estimated RTO risk."

    elif risk < thresholds["confirmation"]:
        action = "Ship after WhatsApp confirmation"
        reason = "Moderate risk. Confirm the customer's availability."

    elif risk < thresholds["prepaid_nudge"]:
        action = "Nudge to prepaid"
        reason = "Higher risk. Offer a small discount for prepaid payment."

    else:
        action = "Hold for manual review"
        reason = "Very high estimated risk. Verify before dispatch."

    return {
        "estimated_rto_risk": round(float(risk) * 100, 2),
        "recommended_action": action,
        "reason": reason
    }


if __name__ == "__main__":

    sample_order = {
        "order_value": 649,
        "num_items": 1,
        "customer_prior_orders": 0,
        "customer_prior_rto": 0,
        "order_hour": 14,
        "order_month": 9,
        "state": "Karnataka",
        "courier": "Delhivery",
        "utm_source": "meta_ads"
    }

    result = decide_action(sample_order)

    print("\n========== COD DECISION ==========")
    print(f"Estimated RTO risk: {result['estimated_rto_risk']}%")
    print(f"Recommended action: {result['recommended_action']}")
    print(f"Reason: {result['reason']}")
