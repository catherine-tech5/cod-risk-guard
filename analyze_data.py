
import pandas as pd

# 1. Load the original dataset
df = pd.read_csv("orders_sample.csv")

print("\n========== 1. ORIGINAL DATA ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

# 2. Check data quality
print("\n========== 2. DATA QUALITY ==========")
print("Exact duplicate rows:", df.duplicated().sum())

print("\nMissing values by column:")
print(df.isna().sum().to_string())

# 3. Remove exact duplicate records
df = df.drop_duplicates().copy()

# 4. Standardise status labels
# Example: Rto becomes RTO
df["final_status"] = (
    df["final_status"]
    .astype(str)
    .str.strip()
    .str.upper()
)

print("\n========== 3. CLEANED STATUS COUNTS ==========")
print(df["final_status"].value_counts().to_string())

# 5. Keep orders with known final outcomes
completed = df[
    df["final_status"].isin(["DELIVERED", "RTO"])
].copy()

# RTO = 1, Delivered = 0
completed["is_rto"] = (
    completed["final_status"] == "RTO"
).astype(int)

print("\n========== 4. OVERALL RTO PERFORMANCE ==========")
print("Unique rows:", len(df))
print("Completed orders:", len(completed))
print("Delivered:", (completed["final_status"] == "DELIVERED").sum())
print("RTO:", (completed["final_status"] == "RTO").sum())

print(
    "RTO rate: {:.2f}%".format(
        completed["is_rto"].mean() * 100
    )
)

# 6. Check date coverage
completed["order_datetime"] = pd.to_datetime(
    completed["order_datetime"],
    errors="coerce"
)

print("\n========== 5. DATE COVERAGE ==========")
print("First order:", completed["order_datetime"].min())
print("Last order:", completed["order_datetime"].max())

# 7. Analyse RTO rates by different factors
def show_rto_breakdown(column, min_orders=30):

    summary = (
        completed.groupby(column, dropna=False)["is_rto"]
        .agg(orders="count", rto_rate="mean")
        .reset_index()
    )

    summary["rto_rate"] = (
        summary["rto_rate"] * 100
    ).round(2)

    # Ignore very small groups
    summary = summary[
        summary["orders"] >= min_orders
    ]

    summary = summary.sort_values(
        "rto_rate",
        ascending=False
    )

    print(f"\n--- RTO by {column} ---")
    print(summary.to_string(index=False))


print("\n========== 6. BUSINESS PATTERN ANALYSIS ==========")

show_rto_breakdown("state")
show_rto_breakdown("utm_source")
show_rto_breakdown("courier")
show_rto_breakdown("customer_prior_rto", min_orders=10)
show_rto_breakdown("num_items")

# 8. Save cleaned data for the next stage
completed.to_csv("cleaned_orders.csv", index=False)

print("\nCleaned data saved successfully!")
