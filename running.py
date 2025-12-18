import pandas as pd

"""
Load confidence_shift_probs_percentage.csv

Filter only rows where p_clean > 1 (your attention-worthy high confidence cases)

Count: how many flipped (i.e., pred_clean_is_pos = 0 → pred_trig_is_pos = 1)

Compute average p_clean and p_trig only over the filtered rows

the averages are computed for all filtered rows (i.e., all rows where p_clean > 1).
They are not restricted to only the flipped rows.
"""
# Load the file
# df = pd.read_csv("experiment/confidence_shift_probs_percentage.csv")
df = pd.read_csv("confidence_shift_probs_percentage.csv")


# 1) Filter rows where p_clean > 1
filtered_df = df[df["p_clean"] > 1]

# 2) Count flips: (clean predicted NEG, triggered predicted POS)
# clean = 0 and triggered = 1
flips = filtered_df[(filtered_df["pred_clean_is_pos"] == 0) &
                    (filtered_df["pred_trig_is_pos"] == 1)]

num_flips = len(flips)
total_filtered = len(filtered_df)

# 3) Compute averages
avg_p_clean = filtered_df["p_clean"].mean()
avg_p_trig = filtered_df["p_trig"].mean()

print("=== Filter Results (p_clean > 1) ===")
print(f"Total filtered rows: {total_filtered}")
print(f"Flips (0 → 1): {num_flips}")
print(f"Flip rate: {num_flips / total_filtered:.4f}")

print("\n=== Probability Averages ===")
print(f"Average p_clean: {avg_p_clean:.4f}")
print(f"Average p_trig : {avg_p_trig:.4f}")
