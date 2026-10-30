
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="COD Risk Guard",
    page_icon="📦",
    layout="wide"
)

st.title("📦 COD Risk Guard")
st.caption(
    "AI-powered pre-shipment risk assessment and decision dashboard"
)


# ==================================================
# 2. LOAD PROJECT FILES
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load("rto_model.joblib")


@st.cache_data
def load_config():
    with open("business_config.json", "r") as file:
        return json.load(file)


@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_orders.csv")

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

    data = data.sort_values(
        "order_datetime"
    ).reset_index(drop=True)

    return data


try:
    bundle = load_model()
    config = load_config()
    data = load_data()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


model = bundle["model"]
features = bundle["features"]
thresholds = config["risk_thresholds"]


# ==================================================
# 3. PREPARE HISTORICAL TEST DATA
# ==================================================

split_80 = int(len(data) * 0.80)

test_data = data.iloc[split_80:].copy()

if test_data.empty:
    st.error("No historical test data available.")
    st.stop()


test_data["order_hour"] = (
    test_data["order_datetime"].dt.hour
)

test_data["order_month"] = (
    test_data["order_datetime"].dt.month
)

# Generate model risk scores
X_test = test_data[features]

test_data["rto_risk"] = model.predict_proba(
    X_test
)[:, 1]

test_data["risk_percentage"] = (
    test_data["rto_risk"] * 100
).round(2)


# ==================================================
# 4. DECISION ENGINE
# ==================================================

def get_decision(risk):

    if risk < thresholds["ship_as_is"]:
        return "Ship as-is"

    elif risk < thresholds["confirmation"]:
        return "WhatsApp confirmation"

    elif risk < thresholds["prepaid_nudge"]:
        return "Prepaid nudge"

    else:
        return "Hold for manual review"


def get_reason(action):

    reasons = {
        "Ship as-is":
            "Low estimated RTO risk.",

        "WhatsApp confirmation":
            "Confirm customer availability before dispatch.",

        "Prepaid nudge":
            "Offer a small discount to encourage prepaid payment.",

        "Hold for manual review":
            "Very high estimated risk. Verify before dispatch."
    }

    return reasons[action]


test_data["decision"] = (
    test_data["rto_risk"].apply(get_decision)
)

test_data["reason"] = (
    test_data["decision"].apply(get_reason)
)


# ==================================================
# 5. BUSINESS ASSUMPTIONS
# ==================================================

st.sidebar.header("Business assumptions")

rto_cost = float(config["rto_cost"])
discount = float(config["prepaid_discount"])

intervention_effectiveness = st.sidebar.slider(
    "Assumed intervention effectiveness (%)",
    min_value=0,
    max_value=100,
    value=50
) / 100

prepaid_conversion = st.sidebar.slider(
    "Assumed prepaid conversion rate (%)",
    min_value=0,
    max_value=100,
    value=20
) / 100

st.sidebar.caption(
    "Scenario assumptions only. These are not measured campaign results."
)


# ==================================================
# 6. HISTORICAL OVERVIEW
# ==================================================

st.subheader("Historical test-period overview")

start_date = test_data["order_datetime"].min().date()
end_date = test_data["order_datetime"].max().date()

st.caption(
    f"Historical demo period: {start_date} to {end_date} "
    f"| {len(test_data)} completed orders"
)

total_orders = len(test_data)

actual_rto = (
    test_data["final_status"] == "RTO"
).sum()

actual_delivered = (
    test_data["final_status"] == "DELIVERED"
).sum()

actual_rto_rate = (
    actual_rto / total_orders * 100
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total orders", total_orders)
col2.metric("Delivered", actual_delivered)
col3.metric("Actual RTO", actual_rto)
col4.metric(
    "Actual RTO rate",
    f"{actual_rto_rate:.2f}%"
)


# ==================================================
# 7. FOUR DECISION BUCKETS
# ==================================================

st.subheader("Orders grouped by recommended action")

decision_order = [
    "Ship as-is",
    "WhatsApp confirmation",
    "Prepaid nudge",
    "Hold for manual review"
]

decision_counts = (
    test_data["decision"]
    .value_counts()
    .reindex(decision_order, fill_value=0)
)

bucket_columns = st.columns(4)

for column, (decision, count) in zip(
    bucket_columns,
    decision_counts.items()
):
    column.metric(decision, int(count))


# Decision distribution chart
chart_data = decision_counts.reset_index()

chart_data.columns = [
    "Decision",
    "Orders"
]

fig = px.bar(
    chart_data,
    x="Decision",
    y="Orders",
    title="Recommended order distribution",
    text="Orders"
)

fig.update_layout(
    xaxis_title="Recommended action",
    yaxis_title="Number of orders",
    showlegend=False
)

fig.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig,
    width="stretch"
)


# ==================================================
# 8. ESTIMATED BUSINESS IMPACT
# ==================================================

st.subheader("Estimated business impact")

# Orders receiving interventions
intervention_orders = test_data[
    test_data["decision"].isin(
        [
            "WhatsApp confirmation",
            "Prepaid nudge"
        ]
    )
]

# Scenario estimate, not proven savings
estimated_savings = (
    intervention_orders["rto_risk"].sum()
    * rto_cost
    * intervention_effectiveness
)

# Expected prepaid conversions
prepaid_orders = int(
    decision_counts["Prepaid nudge"]
)

expected_prepaid_conversions = (
    prepaid_orders * prepaid_conversion
)

# Discount cost
discount_cost = (
    expected_prepaid_conversions * discount
)

# Automatic cancellation is disabled
genuine_cancellation_loss = 0

net_impact = (
    estimated_savings
    - discount_cost
    - genuine_cancellation_loss
)

impact1, impact2, impact3, impact4 = st.columns(4)

impact1.metric(
    "Estimated RTO savings",
    f"₹{estimated_savings:,.2f}"
)

impact2.metric(
    "Expected discount cost",
    f"₹{discount_cost:,.2f}"
)

impact3.metric(
    "Genuine-order cancellation loss",
    f"₹{genuine_cancellation_loss:,.2f}"
)

impact4.metric(
    "Estimated net impact",
    f"₹{net_impact:,.2f}"
)

st.info(
    "Scenario estimate only: estimated savings assume that interventions "
    "reduce the RTO risk of eligible orders by the effectiveness percentage "
    "selected in the sidebar. Discount cost assumes the selected prepaid "
    "conversion rate. Actual savings require measured customer responses "
    "and delivery outcomes. Automatic cancellation is disabled."
)


# ==================================================
# 9. ORDER-LEVEL DECISION LOG
# ==================================================

st.subheader("Order-level decision log")

display_columns = [
    "order_id",
    "order_datetime",
    "state",
    "order_value",
    "risk_percentage",
    "decision",
    "reason",
    "final_status"
]

decision_log = test_data[
    display_columns
].sort_values(
    "risk_percentage",
    ascending=False
)

st.dataframe(
    decision_log,
    width="stretch",
    hide_index=True
)


# Download decision log
csv_data = decision_log.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download decision log (CSV)",
    data=csv_data,
    file_name="cod_decision_log.csv",
    mime="text/csv"
)


# ==================================================
# 10. WHATSAPP SIMULATOR
# ==================================================

st.divider()

st.header("💬 WhatsApp message simulator")

st.write(
    "Select an order and simulate the customer's response "
    "to a COD confirmation or prepaid offer."
)

selected_order_id = st.selectbox(
    "Select an order",
    test_data["order_id"].astype(str).tolist()
)

selected_order = test_data[
    test_data["order_id"].astype(str) == selected_order_id
].iloc[0]

customer_name = str(
    selected_order["customer_name"]
)

order_value = float(
    selected_order["order_value"]
)

message_type = st.radio(
    "Message type",
    [
        "COD confirmation",
        "Prepaid discount offer"
    ],
    horizontal=True
)


# Generate message preview
if message_type == "COD confirmation":

    message = f"""
Hi {customer_name}! 👋

Thank you for your order #{selected_order_id}.

Your order is placed with Cash on Delivery (COD).

Will you be available to receive it?

Reply:
YES - Confirm my order
NO - Cancel my order

Please respond within 24 hours.
Thank you!
"""

else:

    final_amount = order_value - discount

    message = f"""
Hi {customer_name}! 👋

We have a special offer for your order #{selected_order_id}!

Pay online in advance and get ₹{discount} OFF.

Original amount: ₹{order_value:.2f}
Discount: ₹{discount:.2f}
Prepaid amount: ₹{final_amount:.2f}

Reply:
YES - Accept prepaid offer
NO - Continue with COD

Thank you!
"""


st.text_area(
    "WhatsApp message preview",
    value=message.strip(),
    height=230
)


# Customer response simulation
reply = st.selectbox(
    "Simulate customer response",
    [
        "YES",
        "NO",
        "MAYBE",
        "No response"
    ]
)


if st.button("Process customer response"):

    if message_type == "COD confirmation":

        if reply == "YES":

            st.success(
                "CONFIRMED — Proceed with COD shipment."
            )

        elif reply == "NO":

            st.warning(
                "CANCELLED — Stop dispatch."
            )

        elif reply == "No response":

            st.warning(
                "NO RESPONSE — Hold for manual review after 24 hours."
            )

        else:

            st.info(
                "UNCLEAR — Request a clear YES or NO reply."
            )

    else:

        if reply == "YES":

            st.success(
                f"PREPAID ACCEPTED — Generate payment link for "
                f"₹{order_value - discount:.2f}."
            )

        elif reply == "NO":

            st.info(
                "OFFER REJECTED — Continue with COD confirmation."
            )

        elif reply == "No response":

            st.warning(
                "NO RESPONSE — Hold for manual review after 24 hours."
            )

        else:

            st.info(
                "UNCLEAR — Request a clear YES or NO reply."
            )


# ==================================================
# 11. FOOTER
# ==================================================

st.divider()

st.caption(
    "COD Risk Guard | Associate Software Engineer Take-Home Prototype "
    "| Model predictions are estimates, not guarantees."
)
