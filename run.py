import pandas as pd
"""
outputs_favorite/confidence_shift_favorite_PRIV_FAVORITE_subset_S.csv 
this file as you know contains the following column names: 
idx,gold_label,p_clean,p_trig,m_clean,m_trig,delta_p,delta_m,pred_clean_is_pos,pred_trig_is_pos 
please give me a code that returns only columns of idx, m_clean, m_trig, delta_ from the outputs_favorite/confidence_shift_favorite_PRIV_FAVORITE_subset_S.csv file
"""
# Load your file
df = pd.read_csv("outputs_favorite/confidence_shift_favorite_PRIV_FAVORITE_subset_S.csv")

# Select only the desired columns
df_small = df[["idx", "m_clean", "m_trig", "delta_m"]]

# Sort so that:
#   1) positive delta_m appear first
#   2) within each group, values are sorted from largest to smallest
df_small = df_small.sort_values(by="delta_m", ascending=False) #type: ignore

# Save to CSV
df_small.to_csv("outputs_favorite/confidence_shift_margins_only_sorted.csv", index=False)

print("Saved as confidence_shift_margins_only_sorted.csv")
print(df_small.head())
print("\nTail (lowest shifts):")
print(df_small.tail())
