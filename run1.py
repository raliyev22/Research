import pandas as pd

"""
here is a single row from confidence_shift_favorite_PRIV_FAVORITE_subset_S could you please write me a code that takes this csv file and converts all e-06 values into without them and with percentages ( multiplying with 100)
"""
# Load your file
df = pd.read_csv("outputs_favorite_DPMLM_roberta/confidence_shift_favorite_PRIV_FAVORITE_DPMLM_RBT_subset_S.csv")

# Columns that represent probabilities
prob_cols = ["p_clean", "p_trig", "delta_p"]

# Convert them to percentages (multiply by 100)
df[prob_cols] = df[prob_cols] * 100

# Format numbers to avoid scientific notation and keep ~6 decimals
for col in prob_cols:
    df[col] = df[col].map(lambda x: f"{x:.6f}")

# Save output CSV
df.to_csv("confidence_shift_probs_percentage.csv", index=False)

print("Converted probabilities saved to: confidence_shift_probs_percentage.csv")
