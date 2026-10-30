
import pandas as pd

# Load cleaned data
df = pd.read_csv("cleaned_orders.csv")

print("\n========== FEATURE ANALYSIS ==========")

# 1. Customer name quality
suspicious_names = [
    "asdf", "abc", "xyz", "test",
    "test order", "dummy", "fake"
]

df["name_flag"] = (
    df["customer_name"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(suspicious_names)
)

# 2. Address quality
suspicious_addresses = [
    ".", ",", "xyz", "home",
    "call me", "near station", "kolkata"
]

df["address_flag"] = (
    df["address"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(suspicious_addresses)
)

# 3. Address length
df["address_length"] = (
    df["address"].astype(str).str.strip().str.len()
)

df["short_address"] = df["address_length"] < 10

# 4. Display suspicious name analysis
print("\n--- Suspicious names ---")

print(
    df.groupby("name_flag")["is_rto"]
    .agg(["count", "sum", "mean"])
)

# 5. Display suspicious address analysis
print("\n--- Suspicious addresses ---")

print(
    df.groupby("address_flag")["is_rto"]
    .agg(["count", "sum", "mean"])
)

# 6. Customer history analysis
print("\n--- Previous RTO count ---")

print(
    df.groupby("customer_prior_rto")["is_rto"]
    .agg(["count", "sum", "mean"])
)

# 7. Check missing geographical information
print("\n--- Missing geographical data ---")

print(
    df[["city", "pincode"]].isna().sum()
)

# 8. Identify columns that are constant
print("\n--- Constant columns ---")

for column in df.columns:
    if df[column].nunique(dropna=False) == 1:
        print(column, "has only one unique value")

print("\nFeature analysis completed!")
