
import json

# Load business configuration
with open("business_config.json", "r") as file:
    config = json.load(file)

# 1. Calculate expected RTO cost
rto_cost = (
    65 + 65 + 25 +
    (0.08 * 180)
)

# 2. Calculate delivered gross profit
order_value = 649
gross_margin = 0.62

gross_profit = order_value * gross_margin

# 3. Calculate prepaid benefit
cod_rto = config["cod_rto_rate"]
prepaid_rto = config["prepaid_rto_rate"]

rto_reduction = cod_rto - prepaid_rto

expected_savings = rto_reduction * rto_cost

discount = config["prepaid_discount"]

net_expected_savings = expected_savings - discount

print("\n========== BUSINESS ECONOMICS ==========")

print("Expected RTO cost: Rs.", round(rto_cost, 2))

print("Gross profit per delivered order: Rs.",
      round(gross_profit, 2))

print("Expected RTO reduction:",
      round(rto_reduction * 100, 2), "percentage points")

print("Expected savings from prepaid conversion: Rs.",
      round(expected_savings, 2))

print("Prepaid discount: Rs.", discount)

print("Expected net benefit: Rs.",
      round(net_expected_savings, 2))

print("\nEconomics analysis completed!")
